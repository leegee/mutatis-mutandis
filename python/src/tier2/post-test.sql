-- pops
SELECT COUNT(*) FROM tier2_stage.concepts;
SELECT COUNT(*) FROM tier2_stage.concept_field_events;
SELECT COUNT(*) FROM tier2_stage.neighbours;

-- roles
SELECT role, COUNT(*) FROM tier2_stage.concept_field_events GROUP BY role;

-- per-concept size (spot KING vs FANATIC)
SELECT concept, n_events FROM tier2_stage.concepts ORDER BY n_events DESC;

-- no empty concepts
SELECT c.concept FROM tier2_stage.concepts c
LEFT JOIN tier2_stage.concept_field_events f USING (concept)
WHERE f.event_id IS NULL;
