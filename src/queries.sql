-- Reference table defining age brackets used for segment analysis
DROP TABLE IF EXISTS age_bands;
CREATE TABLE age_bands (
    label TEXT,
    min_age INTEGER,
    max_age INTEGER);

INSERT INTO age_bands (label, min_age, max_age) VALUES
    ('18-25', 18, 25),
    ('26-35', 26, 35),
    ('36-45', 36, 45),
    ('46-55', 46, 55),
    ('56-65', 56, 65),
    ('66-110', 66, 110);

-- Dataset size and overall default rate (class imbalance check)
SELECT 
    COUNT(*) AS n, 
    ROUND(AVG(SeriousDlqin2yrs) * 100, 2) AS default_rate 
FROM borrowers;


-- Missing values in MonthlyIncome, with default rate per group.
-- Tests whether missingness is informative: the two rates differ,
-- so the missing flag carries signal and is kept as a feature.
SELECT
    MonthlyIncome IS NULL AS income_is_missing,
    COUNT(*) AS n,
    ROUND(AVG(SeriousDlqin2yrs) * 100, 2) AS default_rate
FROM borrowers
GROUP BY MonthlyIncome IS NULL;


-- Age range sanity check: minimum, maximum, mean and count of records
-- below legal age. Reveals age = 0 used as a placeholder for missing data.
SELECT
    MIN(age) AS minimal_age,
    MAX(age) AS maximal_age,
    AVG(age) AS average_age,
    SUM(age < 18) AS number_of_underage
FROM borrowers;


-- Default rate by age band (JOIN + GROUP BY + HAVING).
-- HAVING filters out groups too small for the rate to be meaningful.
SELECT
    ab.label AS age_band,
    COUNT(*) AS n,
    ROUND(AVG(b.SeriousDlqin2yrs) * 100, 2) AS default_rate
FROM borrowers b
JOIN age_bands ab ON b.age BETWEEN ab.min_age AND ab.max_age
GROUP BY ab.label
HAVING COUNT(*) > 500
ORDER BY default_rate DESC;