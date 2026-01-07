-- inspect liberty in 1641
SELECT rank, neighbour, cosine
FROM neighbourhoods
WHERE query = 'liberty'
  AND slice_start = 1641
ORDER BY rank;
