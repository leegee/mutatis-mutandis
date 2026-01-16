-- candidate semantic anchors
SELECT token, COUNT(DISTINCT doc_id) AS doc_freq
FROM tokens
GROUP BY token
HAVING COUNT(*) > 1000
ORDER BY doc_freq DESC;
