import fs from "node:fs";
import path from "node:path";
import type { Connect } from "vite";
import { json } from "./lib/response";

const XML_CSS_HREF = "/xml-styles/pfs.css";
const TCP_CSS_HREF = "http://www.textcreationpartnership.org/docs/code/pfs.css";

export function createStaticMiddleware(rootDir: string): Connect.NextHandleFunction {
	return (req, res, next) => {
		if (!req.url) return next();

		const match = req.url.match(/^\/(json|xml)\/(.+)$/);
		if (!match) return next();

		const [, type, rawPath] = match;

		/*
		 * XML URLs include the corpus directory in the URL:
		 *
		 *   /xml/eebo_all/eebo_phase2/...
		 *
		 * so the filesystem base is "corpus", not "corpus/eebo_all".
		 */
		const dir = type === "json" ? "out" : "corpus";

		const relativePath = decodeURIComponent(rawPath).replace(/^\/+/, "");

		const rootPath = path.resolve(rootDir);
		const filePath = path.resolve(rootDir, dir, relativePath);

		console.log(`[static] ${type} = ${req.url}\n-> ${filePath}`);

		/*
		 * Prevent path traversal.
		 *
		 * Use a trailing separator so that something like:
		 *
		 *   /root/foo-bar
		 *
		 * cannot incorrectly pass a prefix check against:
		 *
		 *   /root/foo
		 */
		const rootWithSeparator = rootPath.endsWith(path.sep) ? rootPath : rootPath + path.sep;

		if (filePath !== rootPath && !filePath.startsWith(rootWithSeparator)) {
			return json(res, 403, {
				error: "Forbidden path traversal attempt",
				url: req.url,
			});
		}

		if (!fs.existsSync(filePath)) {
			return json(res, 404, {
				error: "File not found",
				url: req.url,
				filePath,
				type,
				dir,
			});
		}

		/*
		 * XML
		 */
		if (type === "xml") {
			res.setHeader("Content-Type", "application/xml; charset=utf-8");

			const stream = fs.createReadStream(filePath, {
				encoding: "utf8",
			});

			let buffer = "";

			stream.on("data", (chunk) => {
				buffer += chunk;
			});

			stream.on("end", () => {
				let rewritten = buffer;

				/*
				 * Replace the TCP stylesheet with our browser stylesheet.
				 */
				rewritten = rewritten.replaceAll(TCP_CSS_HREF, XML_CSS_HREF);

				/*
				 * If the XML doesn't contain an XML stylesheet
				 * processing instruction, add ours.
				 *
				 * We insert it immediately after the XML declaration,
				 * because that is the correct place for an
				 * xml-stylesheet processing instruction.
				 */
				if (!rewritten.includes("<?xml-stylesheet")) {
					const xmlDeclaration = rewritten.match(/^(\uFEFF?<\?xml[^?]*\?>)/);

					if (xmlDeclaration) {
						rewritten =
							rewritten.slice(0, xmlDeclaration[0].length) +
							`\n<?xml-stylesheet type="text/css" href="${XML_CSS_HREF}"?>` +
							rewritten.slice(xmlDeclaration[0].length);
					} else {
						rewritten = `<?xml-stylesheet type="text/css" href="${XML_CSS_HREF}"?>\n` + rewritten;
					}
				}

				res.end(rewritten);
			});

			stream.on("error", (err) => {
				console.error(err);

				if (!res.headersSent) {
					res.statusCode = 500;
					res.end("Internal Server Error");
				}
			});

			return;
		}

		/*
		 * JSON
		 */
		res.setHeader("Content-Type", "application/json; charset=utf-8");

		fs.createReadStream(filePath).pipe(res);
	};
}
