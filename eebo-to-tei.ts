#!/usr/bin/env bun
import fs from "fs";
import path from "path";
import { JSDOM } from "jsdom";
import { create } from "xmlbuilder2";
import { validateTEI } from "./validate-tei";


if (process.argv.length < 3) {
  console.error("Usage: bun eebo-to-tei.ts texts/<basename>");
  process.exit(1);
}

if (!process.argv[2]) throw new Error("Supply an ID - eg 13506_1")

const base = path.basename(process.argv[2]); // e.g., 13506_1
const htmlFile = path.join("texts", `${base}.html`);
const jsonFile = path.join("texts", `${base}.json`);
const outFile = path.join("tei_out", `${base}.xml`);

if (!fs.existsSync(htmlFile)) {
  console.error("HTML file not found:", htmlFile); process.exit(1);
}

const html = fs.readFileSync(htmlFile, "utf-8");
const meta = fs.existsSync(jsonFile)
  ? JSON.parse(fs.readFileSync(jsonFile, "utf-8"))
  : {};

const dom = new JSDOM(html);
const body = dom.window.document.body;

// Create XMLBuilder TEI root
const doc = create({ version: "1.0", encoding: "UTF-8" })
  .ele("TEI", { xmlns: "http://www.tei-c.org/ns/1.0" });

const teiHeader = doc.ele("teiHeader");
const fileDesc = teiHeader.ele("fileDesc");
const titleStmt = fileDesc.ele("titleStmt");
titleStmt.ele("title").txt(meta.title || base);
if (meta.author) titleStmt.ele("author").txt(meta.author);

const publicationStmt = fileDesc.ele("publicationStmt");
publicationStmt.ele("publisher").txt(meta.publisher || "EEBO");
publicationStmt.ele("date").txt(meta.date || "");

// sourceDesc must wrap text in <p>
fileDesc.ele("sourceDesc").ele("p").txt(meta.source || "");

// body must contain a <div type="text">
const textNode = doc.ele("text");
const bodyNode = textNode.ele("body");
const divNode = bodyNode.ele("div", { type: "text" });
body.childNodes.forEach((child) => htmlNodeToTEI(child, divNode));

// Ensure output folder exists
if (!fs.existsSync("tei_out")) fs.mkdirSync("tei_out");

fs.writeFileSync(outFile, doc.end({ prettyPrint: true }));
console.log("Wrote TEI:", outFile);

validateTEI(outFile);



function htmlNodeToTEI(node: Node, parent: any) {
  switch (node.nodeType) {
    case 1: { // Element
      const el = node as HTMLElement;
      let xmlChild: any;

      if (el.tagName === "DIV") {
        xmlChild = parent.ele("div", { type: "text" });
        el.childNodes.forEach((child) => htmlNodeToTEI(child, xmlChild));
        return;
      }

      if (el.tagName === "P") {
        xmlChild = parent.ele("p");
      } else if (el.tagName === "B") {
        xmlChild = parent.node.nodeName === "p" ? parent.ele("hi", { rend: "bold" }) : parent.ele("p").ele("hi", { rend: "bold" });
      } else if (el.tagName === "I") {
        xmlChild = parent.node.nodeName === "p" ? parent.ele("hi", { rend: "italic" }) : parent.ele("p").ele("hi", { rend: "italic" });
      } else {
        // unknown inline content
        xmlChild = parent.node.nodeName === "p" ? parent : parent.ele("p");
      }

      el.childNodes.forEach((child) => htmlNodeToTEI(child, xmlChild));
      break;
    }

    case 3: { // Text
      const text = node.textContent;
      if (text && text.trim().length > 0) {
        if (parent.node.nodeName === "div" || parent.node.nodeName === "body") {
          parent.ele("p").txt(text.replace(/\s+/g, " "));
        } else {
          parent.txt(text.replace(/\s+/g, " "));
        }
      }
      break;
    }
  }
}
