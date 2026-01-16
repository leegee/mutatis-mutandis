-- token frequency by slice
SELECT token, COUNT(*) AS freq
FROM tokens t
JOIN documents d USING (doc_id)
WHERE d.slice_start = 1640
GROUP BY token
ORDER BY freq DESC
LIMIT 100;
