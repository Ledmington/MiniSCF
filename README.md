# MiniSCF

A tiny (Restricted) Hartree-Fock SCF solver.

The project is divided in many small sub-modules. The most important ones are:
- `rhf_sim`: the actual simulator
- `internal_crd`: a converter to/from xyz and internal coordinates
- `bse_get`: a utility executable to download basis set files from the [Basis Set Exchange](https://www.basissetexchange.org/)

## How to use

```bash
cargo build --release
./target/release/rhf_sim --input-xyz my_molecule.xyz --basis-file my_basis.gbs
```

## How to contribute
```bash
cargo build
cargo test
cargo clippy --all-features --all-targets
cargo doc
```
