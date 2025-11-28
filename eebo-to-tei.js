import fs from "fs";
import * as cheerio from "cheerio";
import { htmlDir, htmlFile, jsonFile, normalisePhiloId } from "./lib";
import { validateTEI } from './validate-tei';

const outDir = "./tei_out";
if (!fs.existsSync(outDir)) fs.mkdirSync(outDir);

function xmlEscape(s) {
    if (!s) return "";
    return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function pageNumFromFacs(facs) {
    if (!facs) return null;
    const m = facs.match(/:(\d+)$/);
    if (m) return String(parseInt(m[1], 10));
    return null;
}

function loadAndRepairFragmentHtml(html) {
    const $ = cheerio.load(`<root>${html}</root>`, { decodeEntities: false });

    $('span.xml-pb').each((i, el) => {
        const facs = $(el).attr("facs");
        const pb = `<pb${facs ? ` facs="${xmlEscape(facs)}"` : ""}/>`;
        $(el).replaceWith(pb);
    });

    $('span.xml-hi').each((i, el) => {
        const inner = $(el).html() ?? "";
        $(el).replaceWith(`<hi>${inner}</hi>`);
    });

    $('span.xml-gap').each((i, el) => {
        const inner = $(el).text().trim();
        $(el).replaceWith(inner ? `<gap>${xmlEscape(inner)}</gap>` : `<gap/>`);
    });

    $('span.xml-g').each((i, el) => {
        const txt = $(el).text();
        $(el).replaceWith(xmlEscape(txt));
    });

    $('span').each((i, el) => {
        const classes = ($(el).attr('class') || "").split(/\s+/);
        const known = ['xml-pb', 'xml-hi', 'xml-gap', 'xml-g'];
        if (classes.some(c => known.includes(c))) return;
        const inner = $(el).html() ?? "";
        $(el).replaceWith(inner);
    });

    $('b').each((i, el) => {
        const inner = $(el).html() ?? "";
        $(el).replaceWith(`<hi rend="bold">${inner}</hi>`);
    });
    $('i').each((i, el) => {
        const inner = $(el).html() ?? "";
        $(el).replaceWith(`<hi rend="italic">${inner}</hi>`);
    });

    return $;
}

function fragmentRootToTeiBody($) {
    const root = $('root').first();

    root.find('pb').each((i, pb) => {
        const facs = $(pb).attr('facs') || "";
        const pnum = pageNumFromFacs(facs);
        if (pnum) $(pb).attr('n', pnum);
        if (facs) $(pb).attr('facs', facs);
    });

    root.find('ul').each((i, el) => { $(el).replaceWith(`<list>${$(el).html()}</list>`); });
    root.find('li').each((i, el) => { $(el).replaceWith(`<item>${$(el).html()}</item>`); });

    let bodyInner = root.html() || "";
    bodyInner = bodyInner.replace(/<script[\s\S]*?<\/script>/gi, '');
    bodyInner = bodyInner.replace(/<style[\s\S]*?<\/style>/gi, '');
    return bodyInner;
}

function buildTeiHeaderFromMetadata(metadata) {
    const title = xmlEscape(metadata.title);
    const author = xmlEscape(metadata.author);
    const publisher = xmlEscape(metadata.publisher);
    const pubPlace = xmlEscape(metadata.pub_place);
    const pubDate = xmlEscape(metadata.pub_date || metadata.create_date || metadata.year);
    const extent = xmlEscape(metadata.extent);
    const notes = xmlEscape(metadata.notes);
    const idno = xmlEscape(metadata.idno || metadata.filename);
    const collection = xmlEscape(metadata.collection);

    return `  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>${title}</title>
        ${author ? `<author>${author}</author>` : ""}
      </titleStmt>
      <publicationStmt>
        <publisher>${publisher}</publisher>
        ${pubPlace ? `<pubPlace>${pubPlace}</pubPlace>` : ""}
        ${pubDate ? `<date>${pubDate}</date>` : ""}
      </publicationStmt>
      <sourceDesc>
        <bibl>
          ${title ? `<title>${title}</title>` : ""}
          ${author ? `<author>${author}</author>` : ""}
          ${idno ? `<idno>${idno}</idno>` : ""}
          ${collection ? `<collection>${collection}</collection>` : ""}
          ${extent ? `<extent>${extent}</extent>` : ""}
          ${notes ? `<note>${notes}</note>` : ""}
        </bibl>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <creation>
        ${pubDate ? `<date>${pubDate}</date>` : ""}
      </creation>
    </profileDesc>
  </teiHeader>`;
}

function convertFilesToTei(philoDiv1Id) {
    const jsonPath = jsonFile(philoDiv1Id);
    const htmlPath = htmlFile(philoDiv1Id);

    if (!fs.existsSync(jsonPath)) { console.warn(`JSON missing for ${philoDiv1Id}`); return; }
    if (!fs.existsSync(htmlPath)) { console.warn(`HTML missing for ${philoDiv1Id}`); return; }

    const metadata = JSON.parse(fs.readFileSync(jsonPath, "utf-8")) || {};
    if (!Object.keys(metadata).length) {
        console.error(`No metadata in ${jsonPath}`);
        return;
    }

    let html = fs.readFileSync(htmlPath, "utf-8");
    if (!Object.keys(metadata).length) {
        console.error(`No metadata in ${jsonPath}`);
        return;
    }

    // replace spans with TEI-compliant <hi>
    html = html.replace(/<span[^>]*class="xml-hi"[^>]*>(.*?)<\/span>/g, '<hi>$1</hi>');

    // remove class attributes from divs
    html = html.replace(/<div[^>]*class="[^"]*"([^>]*)>/g, '<div$1>');

    // remove <collection> inside <bibl>
    html = html.replace(/<collection>.*?<\/collection>/gs, '');

    const $ = loadAndRepairFragmentHtml(html);
    const bodyContent = fragmentRootToTeiBody($);
    const header = buildTeiHeaderFromMetadata(metadata);

    const tei = `<?xml version="1.0" encoding="utf-8"?>
<!-- ${jsonPath}
     ${htmlPath}
-->
<TEI xmlns="http://www.tei-c.org/ns/1.0">
${header}
  <text>
    <body>
${bodyContent}
    </body>
  </text>
</TEI>`;

    const xmlPath = `${outDir}/${normalisePhiloId(philoDiv1Id)}.xml`;
    fs.writeFileSync(xmlPath, tei, "utf-8");
    console.log(`Wrote TEI: ${xmlPath}`);

    validateTEI(xmlPath)
}

const philoIds = fs.readdirSync(htmlDir)
    .filter(f => f.endsWith(".json"))
    .map(f => f.replace(/\.json$/, ""));

for (const id of philoIds) {
    convertFilesToTei(id);
    throw new Error("stop");
}
