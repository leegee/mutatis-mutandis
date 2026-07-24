import xml.etree.ElementTree as ET
from collections import Counter

root = ET.parse("law_temporal.graphml")

values = []

for edge in root.iter("{http://graphml.graphdrawing.org/xmlns}edge"):
    for data in edge:
        if data.attrib.get("key") == "d3":
            values.append(float(data.text))

print(len(values))
print(min(values))
print(max(values))
print(sum(values)/len(values))