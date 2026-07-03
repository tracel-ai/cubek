use cubek_test_utils::CatalogEntry;

pub struct QrProblem {
    /// Number of rows (`m`) of the input matrix; must be >= `n`.
    pub m: usize,
    /// Number of columns (`n`) of the input matrix.
    pub n: usize,
}

pub fn problems() -> Vec<CatalogEntry<QrProblem>> {
    vec![
        CatalogEntry::new("128x128", "128x128", QrProblem { m: 128, n: 128 }),
        CatalogEntry::new("512x512", "512x512", QrProblem { m: 512, n: 512 }),
        CatalogEntry::new("1024x1024", "1024x1024", QrProblem { m: 1024, n: 1024 }),
        // Tall-skinny case: the shape TSQR-style panel factorizations target.
        CatalogEntry::new(
            "2048x512",
            "2048x512 (tall-skinny)",
            QrProblem { m: 2048, n: 512 },
        ),
    ]
}
