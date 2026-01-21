Docs
====

Older DOT
--
```bash
cd docs/dot-automated
./run-dot.sh
```

Newer UML
---
```bash
cd docs/uml
./run-uml.sh
```

Install the current PlantUML JAR by `jebbs` from VSC Marketplace into `docs/uml/lib`,
use `ALT`+`D` when editing a `.puml` file to produce an image.

To produe a batch of PNGs alongside the source `*.puml`:

```bash
java -jar lib/plantuml-1.2025.10.jar *.puml
```

To produe a batch of SVGs in the `../out/` directory:

```bash
java -jar lib/plantuml-1.2025.10.jar -tsvg -o ../out *.puml
```
