//! The SCF loop on water/6-31G*: one Fock build and a whole RHF simulation.

mod common;

use criterion::{Criterion, criterion_group, criterion_main};

fn scf(c: &mut Criterion) {
    let water = common::water();
    let mut group = c.benchmark_group("water_6-31gstar");
    group.sample_size(20);
    group.bench_function("two_electron_fock", |b| {
        b.iter(|| water.two_electron_fock())
    });
    group.bench_function("rhf", |b| b.iter(|| water.rhf()));
    group.finish();
}

criterion_group!(benches, scf);
criterion_main!(benches);
