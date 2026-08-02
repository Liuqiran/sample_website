# Valle Dorado image geolocation branch

This branch performs a reproducible visual search against public KartaView imagery.
It downloads the supplied target photo from the referenced Douban topic, queries a grid covering
Valle Dorado in Tlalnepantla de Baz, downloads candidate street-level images, and ranks
matches using SIFT feature correspondence, RANSAC geometric consistency, perceptual hash,
and color histograms.

The GitHub Actions workflow writes its output to `results/report.md`,
`results/results.csv`, and `results/top_matches.jpg`.

A high score is only meaningful when supported by both numerous SIFT matches and a strong
RANSAC inlier ratio. A weak result does not rule out the neighborhood because KartaView
coverage is incomplete.
