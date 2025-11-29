/**
 * Ingests the EEBO-TCP data into SQLite
 */
import fs from 'fs';
import { parse } from 'csv-parse/sync';
import { Database } from "bun:sqlite";

const csvFile = "../eebo-data/eebo-tcp_metadata.csv";
const dbFile = "../eebo-data/eebo-tcp_metadata.sqlite";

// Keywords for likely pamphlets
const pamphletKeywords = [
  "tract", "newsbook", "petition", "declaration", "act", "broadsheet"
];

// Read CSV 
const csvData = fs.readFileSync(csvFile, 'utf-8');
const records = parse(csvData, { columns: true, skip_empty_lines: true });

//  Filter for Civil War era + pamphlet keywords
const filteredRecords = records.filter((r: any) => {
  const year = parseInt(r.year, 10);
  if (isNaN(year) || year < 1640 || year > 1665) return false;
  const titleLower = r.title.toLowerCase();
  return pamphletKeywords.some(kw => titleLower.includes(kw));
});

console.log(`Filtered ${filteredRecords.length} candidate pamphlets from ${records.length} total records.`);

const db = new Database(dbFile);

db.run(`
  CREATE TABLE IF NOT EXISTS eebo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    author TEXT,
    title TEXT,
    year TEXT,
    permalink TEXT,
    philo_div1_id TEXT,
    access TEXT
  )
`);

const insertStmt = db.prepare(`
  INSERT INTO eebo (author, title, year, permalink, access)
  VALUES (?, ?, ?, ?, ?)
`);

db.transaction(() => {
  for (const r of filteredRecords as any[]) {
    insertStmt.run(
      r.author,
      r.title,
      r.year || null,
      r.permalink,
      r.access
    );
  }
})();

console.log(`Inserted ${filteredRecords.length} Civil War–era pamphlets into ${dbFile}`);
