#!/usr/bin/env bun
/**
 * eebo-to-tei.ts
 * Convert EEBO HTML fragments (texts/<id>.html) + metadata (texts/<id>.json)
 * into TEI P5, respecting block/inline constraints required by the DTD.
 *
 * Usage:
 *   bun eebo-to-tei.ts <fileOrDir>
 */

import fs from "fs";
import path from "path";
import { JSDOM } from "jsdom";
import { create } from "xmlbuilder2";
import { type XMLBuilder } from "xmlbuilder2/lib/interfaces";
import { validateTEI } from "./validate-tei";
import { EeboDAO } from "./eebo-dao";

const outDir = "../eebo-tei";
if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

interface Metadata {
  filename: string;
  author: string;
  title: string;
  create_date?: string;
  publisher?: string;
  pub_place?: string;
  pub_date?: string;
  extent?: string;
  notes?: string;
  collection?: string;
  idno: string;
  year?: number;
  head?: string;
  type?: string;
  n?: string;
  speaker?: string;
  resp?: string;
  philo_doc_id?: string;
  philo_div1_id?: string;
  philo_div2_id?: string;
  philo_div3_id?: string;
}

if (process.argv.length < 3 || !process.argv[2]) {
  console.error("Usage: bun eebo-to-tei.ts <file or directory>");
  process.exit(1);
}

const dao = new EeboDAO();

const target = process.argv[2];

// Determine files to process
let bases: string[] = [];
if (fs.statSync(target).isDirectory()) {
  bases = fs.readdirSync(target)
    .filter(f => f.endsWith(".html"))
    .map(f => path.basename(f, ".html"));
} else {
  bases = [path.basename(target, ".html")];
}

console.log(`Processing ${bases.length} files...`);

for (const base of bases) {
  try {
    // Skip if already processed successfully
    if (dao.xmlGenerated(base)) {
      console.log("Skipping already processed:", base);
      continue;
    }

    const htmlPath = path.join(target, `${base}.html`);
    const jsonPath = path.join(target, `${base}.json`);
    const outPath = path.join(outDir, `${base}.xml`);

    if (!fs.existsSync(htmlPath)) {
      console.warn("HTML not found:", htmlPath);
      dao.updateStatus(base, 0);
      continue;
    }

    const html = fs.readFileSync(htmlPath, "utf8");
    const meta: Metadata = fs.existsSync(jsonPath)
      ? JSON.parse(fs.readFileSync(jsonPath, "utf8"))
      : { idno: base };

    // Parse HTML fragment (wrap in body)
    const dom = new JSDOM(`<body>${html}</body>`);
    const fragmentRoot = dom.window.document.body;

    // Create TEI skeleton
    const doc = create({ version: "1.0", encoding: "UTF-8" })
      .ele("TEI", { xmlns: "http://www.tei-c.org/ns/1.0" });

    buildTeiHeader(doc, meta);

    const textNode = doc.ele("text");
    const bodyNode = textNode.ele("body");
    const topDiv = bodyNode.ele("div", { type: "text" });

    // Recursive conversion
    htmlNodeToTEI(fragmentRoot, topDiv, null);

    fs.writeFileSync(outPath, doc.end({ prettyPrint: true }), "utf8");
    console.log("Wrote TEI:", outPath);

    validateTEI(outPath);

    dao.updateStatus(base, 1);

  } catch (err) {
    console.error("Error processing", base, err);
    dao.updateStatus(base, 0);
  }
}

/**
 * Recursive HTML → TEI converter (inline/block)
 */
function htmlNodeToTEI(node: Node, blockParent: any, curP: any): { blockParent: any; curP: any } {
  function ensureP() {
    if (curP) return curP;
    curP = blockParent.ele("p");
    return curP;
  }

  if (node.nodeType === 3) { // TEXT
    const text = (node.textContent || "").replace(/\s+/g, " ").trim();
    if (!text) return { blockParent, curP };
    curP = curP || blockParent.ele("p");
    curP.txt(text);
    return { blockParent, curP };
  }

  if (node.nodeType !== 1) return { blockParent, curP };

  const el = node as HTMLElement;
  const tag = el.tagName.toLowerCase();

  if (["div", "section", "article", "body"].includes(tag)) {
    curP = null;
    const newDiv = blockParent.ele("div", { type: "text" });
    for (const child of Array.from(el.childNodes)) {
      const res = htmlNodeToTEI(child, newDiv, curP);
      curP = res.curP;
    }
    return { blockParent, curP: null };
  }

  if (tag === "pb" || (tag === "span" && (el.className || "").includes("xml-pb"))) {
    curP = null;
    const facs = el.getAttribute("facs") || el.getAttribute("data-facs");
    if (facs) blockParent.ele("pb", { facs });
    else blockParent.ele("pb");
    return { blockParent, curP };
  }

  if (tag === "p" || tag === "para") {
    curP = blockParent.ele("p");
    for (const child of Array.from(el.childNodes)) {
      const res = htmlNodeToTEI(child, blockParent, curP);
      curP = res.curP;
    }
    return { blockParent, curP: null };
  }

  if (/^h[1-6]$/.test(tag)) {
    curP = null;
    const head = blockParent.ele("head");
    head.txt(el.textContent?.replace(/\s+/g, " ").trim() || "");
    return { blockParent, curP };
  }

  if (["b", "strong"].includes(tag)) {
    const p = ensureP();
    const hi = p.ele("hi", { rend: "bold" });
    for (const child of Array.from(el.childNodes)) htmlNodeToTEI(child, blockParent, hi);
    return { blockParent, curP: p };
  }

  if (["i", "em"].includes(tag)) {
    const p = ensureP();
    const hi = p.ele("hi", { rend: "italic" });
    for (const child of Array.from(el.childNodes)) htmlNodeToTEI(child, blockParent, hi);
    return { blockParent, curP: p };
  }

  if (tag === "a") {
    const p = ensureP();
    const href = el.getAttribute("href");
    const ref = href ? p.ele("ref", { target: href }) : p.ele("ref");
    for (const child of Array.from(el.childNodes)) htmlNodeToTEI(child, blockParent, ref);
    return { blockParent, curP: p };
  }

  if (tag === "br") {
    const p = ensureP();
    p.ele("lb");
    return { blockParent, curP };
  }

  if (["ul", "ol"].includes(tag)) {
    curP = null;
    const list = blockParent.ele("list");
    for (const li of Array.from(el.querySelectorAll("li"))) {
      const item = list.ele("item");
      for (const child of Array.from(li.childNodes)) htmlNodeToTEI(child, item, null);
    }
    return { blockParent, curP: null };
  }

  if (["span", "emph", "small", "strong"].includes(tag)) {
    const p = ensureP();
    for (const child of Array.from(el.childNodes)) htmlNodeToTEI(child, blockParent, p);
    return { blockParent, curP: p };
  }

  // default fallback: recurse over children
  for (const child of Array.from(el.childNodes)) {
    const res = htmlNodeToTEI(child, blockParent, curP);
    blockParent = res.blockParent;
    curP = res.curP;
  }

  return { blockParent, curP };
}

/**
 * Build TEI Header (unchanged)
 */
export function buildTeiHeader(root: XMLBuilder, meta: Metadata) {
  const header = root.ele("teiHeader");
  const fileDesc = header.ele("fileDesc");

  const titleStmt = fileDesc.ele("titleStmt");
  titleStmt.ele("title").txt(meta.title || "Untitled");
  if (meta.author) titleStmt.ele("author").txt(meta.author);

  const pub = fileDesc.ele("publicationStmt");
  if (meta.publisher) pub.ele("publisher").txt(meta.publisher);
  else pub.ele("authority").txt("Unknown");

  if (meta.pub_place) pub.ele("pubPlace").txt(meta.pub_place);

  if (meta.pub_date || meta.create_date) {
    const dateStr = meta.pub_date || meta.create_date;
    const year = meta.year || parseInt((dateStr || "").replace(/\D/g, "")) || undefined;
    if (year) pub.ele("date", { when: year.toString() }).txt(dateStr || "");
    else pub.ele("date").txt(dateStr || "");
  }

  if (meta.idno) pub.ele("idno", { type: "eebo" }).txt(meta.idno);

  pub.ele("availability").ele("p").txt("Digitised for research use.");

  const sourceDesc = fileDesc.ele("sourceDesc");
  const bibl = sourceDesc.ele("bibl");
  bibl.ele("title").txt(meta.title || "Untitled");
  if (meta.author) bibl.ele("author").txt(meta.author);
  if (meta.extent) bibl.ele("extent").txt(meta.extent);
  if (meta.notes) {
    bibl.ele("note").txt(meta.notes);
    bibl.ele("note", { type: "eebo-tcp" }).txt(meta.notes);
  }
  if (meta.collection) bibl.ele("idno", { type: "collection" }).txt(meta.collection);
}
