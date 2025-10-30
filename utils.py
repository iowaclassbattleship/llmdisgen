def write_txt(filename, lines):
    with open(filename, "w") as f:
        f.writelines(lines)