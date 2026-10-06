// Clarabel.rs comparison driver, run out-of-tree from PureOSQP.jl's benchmark harness.
//
// Reads one problem per `<data_dir>/*.txt` file (CSC `P`/`A` plus `q`/`b`, one-sided
// cones, written by ../clarabel_rs_compare.jl in the same format `run_problem` below
// parses) and times `DefaultSolver::new` + `solve()` with this process's own clock.
// The minimum over repeated solves within a wall-clock budget is reported, matching
// Chairmarks' `@b` (field-wise minimum) used for the Julia solvers in this comparison.
//
// Build (out-of-tree target dir, per the no-Rust-in-the-repo rule for PureOSQP.jl):
//     cargo build --release --manifest-path bench/clarabel_rs/Cargo.toml \
//         --target-dir /tmp/clarabel_rs_target
// Run (single-threaded, pinned core):
//     RAYON_NUM_THREADS=1 taskset -c 15 \
//         /tmp/clarabel_rs_target/release/clarabel_rs_bench <data_dir> [seconds] [tol] [ldl]
//
// `ldl` names Clarabel's `direct_solve_method`: "faer" (the default here) or "qdldl". Both are
// sparse LDL factorizations of the same KKT matrix, so the pair separates the factorization
// from everything else Clarabel does. faer switches from a simplicial factorization to a
// supernodal one — which blocks the work into dense kernels — above a flops/nnz(L) ratio of
// 40, and Clarabel hardcodes that `AUTO` threshold, so which path a case takes is a property
// of the case rather than a setting. `solver.info.linsolver.name` is reported per case and
// names the method that ran.
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, NonnegativeConeT, SupportedConeT,
};
use serde::Serialize;

struct Problem {
    n: usize,
    m: usize,
    p_colptr: Vec<usize>,
    p_rowval: Vec<usize>,
    p_nzval: Vec<f64>,
    q: Vec<f64>,
    a_colptr: Vec<usize>,
    a_rowval: Vec<usize>,
    a_nzval: Vec<f64>,
    b: Vec<f64>,
}

fn parse_ints(line: &str) -> Vec<usize> {
    line.split_whitespace().map(|s| s.parse().unwrap()).collect()
}

fn parse_floats(line: &str) -> Vec<f64> {
    line.split_whitespace().map(|s| s.parse().unwrap()).collect()
}

/// Line format: header `n m nnzP nnzA`, then eight lines of whitespace-separated
/// numbers: `Pcolptr Prowval Pnzval q Acolptr Arowval Anzval b` (0-based indices,
/// `P` upper-triangular, `A`/`b` already in one-sided `Ax <= b` form).
fn read_problem(path: &Path) -> Problem {
    let file = fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = BufReader::new(file).lines().map(|l| l.unwrap());
    let mut next = || lines.next().expect("truncated problem file");

    let header = parse_ints(&next());
    let (n, m, nnz_p, nnz_a) = (header[0], header[1], header[2], header[3]);

    let p_colptr = parse_ints(&next());
    let p_rowval = parse_ints(&next());
    let p_nzval = parse_floats(&next());
    let q = parse_floats(&next());
    let a_colptr = parse_ints(&next());
    let a_rowval = parse_ints(&next());
    let a_nzval = parse_floats(&next());
    let b = parse_floats(&next());

    assert_eq!(p_nzval.len(), nnz_p, "P nzval count mismatch");
    assert_eq!(a_nzval.len(), nnz_a, "A nzval count mismatch");
    Problem { n, m, p_colptr, p_rowval, p_nzval, q, a_colptr, a_rowval, a_nzval, b }
}

#[derive(Serialize)]
struct CaseResult {
    name: String,
    n: usize,
    m: usize,
    iterations: u32,
    status: String,
    obj_val: f64,
    /// Minimum of `DefaultSolver::new` + `solve()`, this process's own `Instant` clock,
    /// over `reps` repeats within the wall-clock budget. Excludes file I/O and the one
    /// Cargo/OS process start paid once for the whole run.
    solve_time_self_s: f64,
    reps: usize,
    linsolver: String,
    x: Vec<f64>,
}

fn run_problem(path: &Path, seconds: f64, tol: f64, ldl: &str) -> CaseResult {
    let name = path.file_stem().unwrap().to_string_lossy().into_owned();
    let prob = read_problem(path);

    let p = CscMatrix::new(prob.n, prob.n, prob.p_colptr, prob.p_rowval, prob.p_nzval);
    let a = CscMatrix::new(prob.m, prob.n, prob.a_colptr, prob.a_rowval, prob.a_nzval);
    let cones: Vec<SupportedConeT<f64>> = vec![NonnegativeConeT(prob.m)];

    // The method is named rather than left on "auto", which selects between Clarabel's own
    // backends by availability and would hide which factorization ran.
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .tol_gap_abs(tol)
        .tol_gap_rel(tol)
        .tol_feas(tol)
        .direct_solve_method(ldl.to_string())
        .max_threads(1)
        .build()
        .expect("invalid Clarabel settings");

    let budget = Duration::from_secs_f64(seconds);
    let budget_start = Instant::now();
    let mut best = f64::INFINITY;
    let mut reps = 0usize;
    // Overwritten every iteration below; only the last (fastest-budget-exceeding) solve's
    // values are kept, since Clarabel's iterate is deterministic across repeats.
    #[allow(unused_assignments)]
    let (mut linsolver, mut iterations, mut status, mut obj_val, mut x) =
        (String::new(), 0u32, String::new(), f64::NAN, Vec::new());

    loop {
        let t0 = Instant::now();
        let mut solver = DefaultSolver::new(&p, &prob.q, &a, &prob.b, &cones, settings.clone())
            .expect("Clarabel setup failed");
        solver.solve();
        let elapsed = t0.elapsed().as_secs_f64();
        reps += 1;
        if elapsed < best {
            best = elapsed;
        }
        linsolver = solver.info.linsolver.name.clone();
        iterations = solver.solution.iterations;
        status = format!("{:?}", solver.solution.status);
        obj_val = solver.solution.obj_val;
        x = solver.solution.x.clone();
        if budget_start.elapsed() >= budget {
            break;
        }
    }

    eprintln!(
        "{name:<10} n={n:<4} m={m:<5} | {iterations:3} it {ms:8.3} ms | linsolver={linsolver} reps={reps}",
        n = prob.n, m = prob.m, ms = 1.0e3 * best,
    );

    CaseResult { name, n: prob.n, m: prob.m, iterations, status, obj_val, solve_time_self_s: best, reps, linsolver, x }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: clarabel_rs_bench <data_dir> [seconds=0.3] [tol=1e-8] [ldl=faer]");
        std::process::exit(2);
    }
    let data_dir = &args[1];
    let seconds: f64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(0.3);
    let tol: f64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1.0e-8);
    let ldl = args.get(4).map(String::as_str).unwrap_or("faer");

    let mut files: Vec<PathBuf> = fs::read_dir(data_dir)
        .unwrap_or_else(|e| panic!("{data_dir}: {e}"))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().map_or(false, |e| e == "txt"))
        .collect();
    files.sort();

    let results: Vec<CaseResult> =
        files.iter().map(|p| run_problem(p, seconds, tol, ldl)).collect();
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}
