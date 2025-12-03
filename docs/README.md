Docs
====

Install PlantUML by `jebbs` from VSC Marketplace, `ALT`+`D` to produce an image. 

To produe a batch of PNGs alongside the source `*.puml`:

```bash
java -jar lib/plantuml-1.2025.10.jar *.puml 
```

To produe a batch of SVGs in the `../output/` directory:

```bash
java -jar lib/plantuml-1.2025.10.jar -tsvg -o ../output *.puml
```
