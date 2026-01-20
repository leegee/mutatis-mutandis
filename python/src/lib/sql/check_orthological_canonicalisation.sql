-- 1. Check for false positives accidentally mapped
SELECT sm.variant, sm.canonical, k.false_positives
FROM spelling_map sm
JOIN (
    VALUES
        ('justice', ARRAY['injury','injustice']),
        ('injustice', ARRAY['injury'])
) AS k(canonical, false_positives)
ON sm.canonical = k.canonical
WHERE sm.variant = ANY(k.false_positives);

-- 2. Check that all known allowed variants are present
SELECT v.canonical, v.allowed_variant
FROM (
    VALUES
        ('justice', 'unjustice'),
        ('justice', 'vnjustice'),
        ('justice', 'dinjustice'),
        ('justice', 'iujustice'),
        ('justice', 'chiefjustice'),
        ('justice', 'executejustice'),
        ('justice', 'satisfiedjustice'),
        ('injustice', 'dojustice'),
        ('injustice', 'ofjustice')
) AS v(canonical, allowed_variant)
LEFT JOIN spelling_map sm
    ON sm.canonical = v.canonical
   AND sm.variant = v.allowed_variant
WHERE sm.variant IS NULL;
