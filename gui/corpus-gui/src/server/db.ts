import postgres, { type Sql } from "postgres";

let sql: Sql | undefined;

export function get_connection(): Sql {
	if (sql) return sql;

	sql = postgres({
		max: 10,
		idle_timeout: 20,
		connect_timeout: 10,
	});

	return sql;
}
