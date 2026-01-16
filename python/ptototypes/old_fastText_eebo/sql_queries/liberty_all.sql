-- compare liberty across slices
SELECT *
FROM neighbourhoods
WHERE query = 'liberty'
  AND rank <= 10
-- ORDER BY slice_start, rank;
ORDER BY rank, slice_start
