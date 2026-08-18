import Ajv2020, { type ErrorObject, } from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { parse } from "jsonc-parser";

import type { ResearchProject } from "./project";

import schemaText from "./research-project.schema.jsonc?raw";

const schema = parse(schemaText);

const ajv = new Ajv2020({
    allErrors: true,
    strict: true,
});

addFormats(ajv);

const validate = ajv.compile<ResearchProject>(
    schema
);

export interface ProjectValidationError {
    path: string;
    message: string;
}

export interface ProjectValidationResult {
    valid: boolean;
    errors: ProjectValidationError[];
}

function formatPath(error: ErrorObject): string {
    if (error.instancePath) {
        return error.instancePath
            .replace(/\//g, ".")
            .replace(/^\./, "");
    }

    if (error.keyword === "required") {
        const missingProperty =
            typeof error.params === "object" &&
                error.params !== null &&
                "missingProperty" in error.params
                ? String(error.params.missingProperty)
                : "";

        return missingProperty
            ? `${ error.instancePath || "project" }.${ missingProperty }`
            : error.instancePath || "project";
    }

    return error.instancePath || "project";
}

function formatError(
    error: ErrorObject,
): ProjectValidationError {
    const path = formatPath(error);

    switch (error.keyword) {
        case "required":
            return {
                path,
                message: error.message ?? "is required",
            };

        case "additionalProperties":
            return {
                path,
                message:
                    `unexpected property "${ String(
                        error.params.additionalProperty,
                    ) }"`,
            };

        case "enum":
            return {
                path,
                message:
                    `${ error.message }: ${ (
                        error.params as {
                            allowedValues: unknown[];
                        }
                    ).allowedValues.join(", ") }`,
            };

        case "const":
            return {
                path,
                message:
                    `must equal ${ JSON.stringify(
                        error.params.allowedValue,
                    ) }`,
            };

        default:
            return {
                path,
                message: error.message ?? "invalid value",
            };
    }
}


export function validateProject(
    value: unknown,
): ProjectValidationResult {
    const valid = validate(value);

    if (!valid) {
        return {
            valid: false,
            errors: (validate.errors ?? [])
                .map(formatError),
        };
    }

    const project = value as ResearchProject;

    const entityIds = new Set(
        project.entities.map(
            (entity) => entity.id,
        ),
    );

    const errors: ProjectValidationError[] = [];

    for (const relation of project.relations) {
        if (!entityIds.has(relation.sourceId)) {
            errors.push({
                path: `relations.${ relation.id }.sourceId`,
                message:
                    `references missing entity "${ relation.sourceId }"`,
            });
        }

        if (!entityIds.has(relation.targetId)) {
            errors.push({
                path: `relations.${ relation.id }.targetId`,
                message:
                    `references missing entity "${ relation.targetId }"`,
            });
        }
    }

    const relationIds = new Set(
        project.relations.map(
            (relation) => relation.id,
        ),
    );

    for (const evidence of project.evidence) {
        for (const entityId of evidence.entityIds) {
            if (!entityIds.has(entityId)) {
                errors.push({
                    path: `evidence.${ evidence.id }.entityIds`,
                    message:
                        `references missing entity "${ entityId }"`,
                });
            }
        }

        for (const relationId of evidence.relationIds) {
            if (!relationIds.has(relationId)) {
                errors.push({
                    path: `evidence.${ evidence.id }.relationIds`,
                    message: `references missing relation "${ relationId }"`,
                });
            }
        }
    }

    return {
        valid: errors.length === 0,
        errors,
    };
}


export function isValidProject(
    value: unknown,
): value is ResearchProject {
    return validate(value);
}
