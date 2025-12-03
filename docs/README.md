Docs
====

DOT
--
```bash
cd docs/dot-automated
./run-dot.sh
```

UML
---
```bash
cd docs/uml
./run-uml.sh
```

Install from VSC Marketplace into `docs/uml/lib` the current PlantUML JAR by `jebbs`, 
use `ALT`+`D` when editing a `.puml` file to produce an image. 

To produe a batch of PNGs alongside the source `*.puml`:

```bash
java -jar lib/plantuml-1.2025.10.jar *.puml 
```

To produe a batch of SVGs in the `../output/` directory:

```bash
java -jar lib/plantuml-1.2025.10.jar -tsvg -o ../output *.puml
```
