// db.test.ts
import { describe, expect, test } from "bun:test";
import { get_connection } from "./db";

describe("DB", () => {
	test("can access the events table", async () => {
		const db = get_connection();

		const rows = await db`SELECT COUNT(*)::integer AS count FROM events `;

		expect(rows).toHaveLength(1);
		expect(rows[0]?.count).toBeGreaterThanOrEqual(0);
	});
});
