# find_open_xml_tags.py
import sys, xml.parsers.expat as X

p = X.ParserCreate()
stack = []
p.StartElementHandler = lambda n, a: stack.append((n, p.CurrentLineNumber))
p.EndElementHandler = lambda n: stack.pop()
try:
    p.Parse(open(sys.argv[1], "rb").read(), True)
    print("OK")
except X.ExpatError as e:
    print(e)
    print("open elements (outermost -> innermost):", stack[-6:])