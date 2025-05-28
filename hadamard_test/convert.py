#!/usr/bin/env python3
import sys
import re

def py2c_hadK_converter(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    state = "SEARCH"           # SEARCH -> LOOKBRACKET -> IN_LITERAL
    func_name = None
    N = None
    literal_lines = []
    bracket_depth = 0

    with open(output_path, 'w') as out:
        for line in lines:
            if state == "SEARCH":
                m = re.match(r"\s*def\s+get_had(\d+)\s*\(", line)
                if m:
                    N = int(m.group(1))
                    func_name = f"get_had{N}"
                    out.write(f"// Converted from Python {func_name}\n")
                    out.write(f"static const float {func_name}_data[{N}][{N}] = {{\n")
                    # initialize literal capture
                    literal_lines = []
                    bracket_depth = 0
                    state = "LOOKBRACKET"
                continue

            if state == "LOOKBRACKET":
                idx = line.find('[')
                if idx >= 0:
                    # Start of literal from first '[' onward
                    literal = line[idx:]
                    bracket_depth = literal.count('[') - literal.count(']')
                    conv = literal.replace('[', '{').replace(']', '}')
                    literal_lines = [conv]
                    state = "IN_LITERAL"
                continue

            if state == "IN_LITERAL":
                # update bracket depth over entire line
                bracket_depth += line.count('[') - line.count(']')
                # convert all brackets
                conv = line.replace('[', '{').replace(']', '}')
                literal_lines.append(conv)

                if bracket_depth == 0:
                                        # emit the full C initializer (skip outer braces)
                    for lit in literal_lines[1:-1]:
                        out.write(lit)
                    out.write("};\n")
                    # emit stub function
                    out.write(f"void {func_name}(float **out_hadK) {{\n")
                    out.write(f"    *out_hadK = (float*){func_name}_data;  // static data\n")
                    out.write("}\n")
                    # reset state
                    state = "SEARCH"
                    func_name = None
                    N = None
                    literal_lines = []
                    bracket_depth = 0

                    state = "SEARCH"
                    func_name = None
                    N = None
                    literal_lines = []
                    bracket_depth = 0
                continue

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: py2c_hadK_converter.py <input_python_file> <output_c_file>")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    py2c_hadK_converter(inp, outp)
