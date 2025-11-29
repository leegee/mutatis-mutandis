#!/usr/bin/env bun
/**
 * eebo-to-tei.ts
 * Convert an EEBO HTML fragment (texts/<id>.html) + metadata (texts/<id>.json)
 * into TEI P5 that respects block/inline constraints required by the DTD.
 *
 * Usage:
 *   bun eebo-to-tei.ts 13506_1
 */

import fs from "fs";
import path from "path";
import { JSDOM } from "jsdom";
import { create } from "xmlbuilder2";
import { validateTEI } from "./validate-tei";

if (process.argv.length < 3 || !process.argv[2]) {
  console.error("Usage: bun eebo-to-tei.ts <baseId>");
  process.exit(1);
}

const base = path.basename(process.argv[2]); // e.g. "13506_1"
const htmlPath = path.join("texts", `${base}.html`);
const jsonPath = path.join("texts", `${base}.json`);
const outDir = "tei_out";
const outPath = path.join(outDir, `${base}.xml`);

if (!fs.existsSync(htmlPath)) {
  console.error("HTML not found:", htmlPath);
  process.exit(1);
}

const html = fs.readFileSync(htmlPath, "utf8");
const meta = fs.existsSync(jsonPath) ? JSON.parse(fs.readFileSync(jsonPath, "utf8")) : {};

// Parse HTML fragment (wrap in a body to be safe)
const dom = new JSDOM(`<body>${html}</body>`);
const document = dom.window.document;
const fragmentRoot = document.body;

// Create TEI skeleton
const doc = create({ version: "1.0", encoding: "UTF-8" })
  .ele("TEI", { xmlns: "http://www.tei-c.org/ns/1.0" });

const teiHeader = doc.ele("teiHeader");
const fileDesc = teiHeader.ele("fileDesc");
const titleStmt = fileDesc.ele("titleStmt");
titleStmt.ele("title").txt((meta.title || "").toString() || base);
if (meta.author) titleStmt.ele("author").txt(meta.author);

const publicationStmt = fileDesc.ele("publicationStmt");
if (meta.publisher) publicationStmt.ele("publisher").txt(meta.publisher);
if (meta.pubPlace) publicationStmt.ele("pubPlace").txt(meta.pubPlace);
if (meta.date) publicationStmt.ele("date").txt(meta.date);

// sourceDesc must contain <p> (or bibl, etc.) per DTD
const sourceDesc = fileDesc.ele("sourceDesc");
sourceDesc.ele("p").txt(meta.source || `EEBO HTML fragment ${base}`);

const textNode = doc.ele("text");
const bodyNode = textNode.ele("body");
// top-level div in body; we'll use this as the initial block parent
const topDiv = bodyNode.ele("div", { type: "text" });

/**
 * Walk the HTML and append TEI to the XML builder.
 *
 * We keep two pointers:
 *  - blockParent: nearest TEI block (a <div> in the TEI body)
 *  - curP: current TEI <p> (for inline content). If null, text/inline will auto-open a <p>.
 *
 * This prevents illegal nesting like <div> inside <p> or <p> inside <hi>.
 */
function htmlNodeToTEI(node: Node, blockParent: any, curP: any): { blockParent: any; curP: any } {
  // Helper: ensure we have a paragraph to append inline content
  function ensureP(): any {
    if (curP) return curP;
    curP = blockParent.ele("p");
    return curP;
  }

  if (node.nodeType === 3) { // TEXT
    const raw = node.textContent || "";
    const text = raw.replace(/\s+/g, " ").trim();
    if (!text) return { blockParent, curP };

    // if we're currently inside a block but not inside a <p>, open one
    if (!curP) {
      curP = blockParent.ele("p");
    }
    curP.txt(text);
    return { blockParent, curP };
  }

  if (node.nodeType !== 1) return { blockParent, curP }; // ignore comments, etc.

  const el = node as HTMLElement;
  const tag = el.tagName.toLowerCase();

  // BLOCK elements
  if (tag === "div" || tag === "section" || tag === "article" || tag === "body") {
    // close current paragraph context (we will not put a <div> inside a <p>)
    curP = null;
    const newDiv = blockParent.ele("div", { type: "text" });
    for (const child of Array.from(el.childNodes)) {
      const res = htmlNodeToTEI(child, newDiv, curP);
      newDiv; // keep
      curP = res.curP;
    }
    // after finishing a block-level element, close paragraph context
    curP = null;
    return { blockParent, curP };
  }

  // Page breaks: TEI <pb/> should be placed at block level (as sibling to <p>)
  if (tag === "pb" || (tag === "span" && (el.className || "").includes("xml-pb"))) {
    // close any current paragraph context (so pb is sibling to p)
    curP = null;
    const facs = (el.getAttribute("facs") || el.getAttribute("data-facs")) || undefined;
    if (facs) blockParent.ele("pb", { facs });
    else blockParent.ele("pb");
    return { blockParent, curP };
  }

  // Paragraph: create <p> under current block parent
  if (tag === "p" || tag === "para") {
    // close any previous paragraph and start a new one
    curP = blockParent.ele("p");
    for (const child of Array.from(el.childNodes)) {
      const res = htmlNodeToTEI(child, blockParent, curP);
      curP = res.curP;
    }
    // keep curP as is (we may add inline content later) or close it
    // we choose to close after finishing the <p>
    curP = null;
    return { blockParent, curP };
  }

  // Headings could map to <head> within blockParent (allowed)
  if (/^h[1-6]$/.test(tag)) {
    // close paragraph context
    curP = null;
    const head = blockParent.ele("head");
    for (const child of Array.from(el.childNodes)) {
      // inside head, we treat children like inline but we can simply append text
      const txt = (child.textContent || "").replace(/\s+/g, " ").trim();
      if (txt) head.txt(txt);
    }
    return { blockParent, curP };
  }

  // Inline formatting — must be inside a <p> (ensure one exists)
  if (tag === "b" || tag === "strong") {
    const p = ensureP();
    const hi = p.ele("hi", { rend: "bold" });
    for (const child of Array.from(el.childNodes)) {
      const res = htmlNodeToTEI(child, blockParent, hi);
      // ensure we don't accidentally set curP to an inline element
      if (res.curP && res.curP !== hi) {
        // if child created a paragraph, keep original p
        // (unlikely since inline children shouldn't create block <p>)
      }
    }
    return { blockParent, curP: p };
  }

  if (tag === "i" || tag === "em") {
    const p = ensureP();
    const hi = p.ele("hi", { rend: "italic" });
    for (const child of Array.from(el.childNodes)) {
      htmlNodeToTEI(child, blockParent, hi);
    }
    return { blockParent, curP: p };
  }

  // links -> TEI <ref target="...">...</ref>
  if (tag === "a") {
    const href = el.getAttribute("href") || undefined;
    const p = ensureP();
    const ref = href ? p.ele("ref", { target: href }) : p.ele("ref");
    for (const child of Array.from(el.childNodes)) {
      htmlNodeToTEI(child, blockParent, ref);
    }
    return { blockParent, curP: p };
  }

  // line break: use TEI <lb/>
  if (tag === "br") {
    // ensure a paragraph exists to hold the lb
    const p = ensureP();
    p.ele("lb");
    return { blockParent, curP };
  }

  // lists: map <ul>/<ol> -> TEI <list> with <item>
  if (tag === "ul" || tag === "ol") {
    curP = null;
    const list = blockParent.ele("list");
    for (const li of Array.from(el.querySelectorAll("li"))) {
      const item = list.ele("item");
      // each li -> item -> p children
      for (const child of Array.from(li.childNodes)) {
        const res = htmlNodeToTEI(child, item, null);
        // if inline produced text, ensure it's inside <p>
        if (res.curP) {
          // move on
        }
      }
    }
    return { blockParent, curP: null };
  }

  // span / inline fallback: treat as inline; ensure p and recurse
  // (many EEBO inline features live in <span class="xml-hi"> etc.)
  if (tag === "span" || tag === "emph" || tag === "small" || tag === "strong") {
    const p = ensureP();
    // don't create extra hi if span just contains text; but we allow nested parsing
    for (const child of Array.from(el.childNodes)) {
      htmlNodeToTEI(child, blockParent, p);
    }
    return { blockParent, curP: p };
  }

  // default: recurse over children, treating unknown tags as transparent inline/block depending on content
  // we must be careful: unknown block-like tags that contain block children should be handled by the top recursion
  // safest fallback: walk children and let children decide
  for (const child of Array.from(el.childNodes)) {
    const res = htmlNodeToTEI(child, blockParent, curP);
    blockParent = res.blockParent;
    curP = res.curP;
  }
  return { blockParent, curP };
}

// Walk fragment children and convert
let state = { blockParent: topDiv, curP: null as any | null };
for (const child of Array.from(fragmentRoot.childNodes)) {
  state = htmlNodeToTEI(child, state.blockParent, state.curP);
}

// Finalize output
if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(outPath, doc.end({ prettyPrint: true }), "utf8");
console.log("Wrote TEI:", outPath);

// Validate
try {
  validateTEI(outPath);
} catch (err) {
  // validateTEI prints errors itself; rethrow so caller sees exit code
  process.exitCode = 1;
  // still exit after showing messages
}
