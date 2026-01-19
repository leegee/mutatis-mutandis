-- 1. Pick a few keywords
SELECT *
FROM spelling_map
WHERE variant IN ('ofjustice', 'njustice', 'iliberty', 'freedoom', 'reasonabl');

-- 2. Check canonical tokens in the tokens table
SELECT token, canonical
FROM tokens
WHERE token IN ('ofjustice', 'njustice', 'iliberty', 'freedoom', 'reasonabl')
LIMIT 20;

-- 3. Spot-check some aggregates, e.g., how many tokens were canonicalised
SELECT canonical, COUNT(*)
FROM tokens
WHERE canonical IS NOT NULL
GROUP BY canonical
ORDER BY COUNT(*) DESC
LIMIT 20;
