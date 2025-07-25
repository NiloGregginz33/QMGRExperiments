#!/usr/bin/env python3

# Python 3 compatible version of reindent tool
# Based on the original by Tim Peters

import tokenize
import os, shutil
import sys

verbose    = 0
recurse    = 0
dryrun     = 0
makebackup = True

def usage(msg=None):
    if msg is not None:
        print(msg, file=sys.stderr)
    print(__doc__, file=sys.stderr)

def errprint(*args):
    sep = ""
    for arg in args:
        sys.stderr.write(sep + str(arg))
        sep = " "
    sys.stderr.write("\n")

def main():
    import getopt
    global verbose, recurse, dryrun, makebackup
    try:
        opts, args = getopt.getopt(sys.argv[1:], "drnvh",
                        ["dryrun", "recurse", "nobackup", "verbose", "help"])
    except getopt.GetoptError as msg:
        usage(msg)
        return
    for o, a in opts:
        if o in ('-d', '--dryrun'):
            dryrun += 1
        elif o in ('-r', '--recurse'):
            recurse += 1
        elif o in ('-n', '--nobackup'):
            makebackup = False
        elif o in ('-v', '--verbose'):
            verbose += 1
        elif o in ('-h', '--help'):
            usage()
            return
    if not args:
        r = Reindenter(sys.stdin)
        r.run()
        r.write(sys.stdout)
        return
    for arg in args:
        check(arg)

def check(file):
    if os.path.isdir(file) and not os.path.islink(file):
        if verbose:
            print("listing directory", file)
        names = os.listdir(file)
        for name in names:
            fullname = os.path.join(file, name)
            if ((recurse and os.path.isdir(fullname) and
                 not os.path.islink(fullname) and
                 not os.path.split(fullname)[1].startswith("."))
                or name.lower().endswith(".py")):
                check(fullname)
        return

    if verbose:
        print("checking", file, "...", end="")
    try:
        f = open(file, encoding='utf-8')
    except IOError as msg:
        errprint("%s: I/O Error: %s" % (file, str(msg)))
        return

    r = Reindenter(f)
    f.close()
    if r.run():
        if verbose:
            print("changed.")
            if dryrun:
                print("But this is a dry run, so leaving it alone.")
        if not dryrun:
            bak = file + ".bak"
            if makebackup:
                shutil.copyfile(file, bak)
                if verbose:
                    print("backed up", file, "to", bak)
            f = open(file, "w", encoding='utf-8')
            r.write(f)
            f.close()
            if verbose:
                print("wrote new", file)
        return True
    else:
        if verbose:
            print("unchanged.")
        return False

def _rstrip(line, JUNK='\n \t'):
    """Return line stripped of trailing spaces, tabs, newlines."""
    i = len(line)
    while i > 0 and line[i-1] in JUNK:
        i -= 1
    return line[:i]

class Reindenter:

    def __init__(self, f):
        self.find_stmt = 1  # next token begins a fresh stmt?
        self.level = 0      # current indent level

        # Raw file lines.
        self.raw = f.readlines()

        # File lines, rstripped & tab-expanded.
        self.lines = [_rstrip(line).expandtabs() + "\n"
                      for line in self.raw]
        self.lines.insert(0, None)
        self.index = 1  # index into self.lines of next line

        # List of (lineno, indentlevel) pairs, one for each stmt and
        # comment line.  indentlevel is -1 for comment lines.
        self.stats = []

    def run(self):
        tokenize.tokenize(self.getline, self.tokeneater)
        # Remove trailing empty lines.
        lines = self.lines
        while lines and lines[-1] == "\n":
            lines.pop()
        # Sentinel.
        stats = self.stats
        stats.append((len(lines), 0))
        # Map count of leading spaces to # we want.
        have2want = {}
        # Program after transformation.
        after = self.after = []
        # Copy over initial empty lines
        i = stats[0][0]
        after.extend(lines[1:i])
        for i in range(len(stats)-1):
            thisstmt, thislevel = stats[i]
            nextstmt = stats[i+1][0]
            have = getlspace(lines[thisstmt])
            want = thislevel * 4
            if want < 0:
                # A comment line.
                if have:
                    want = have2want.get(have, -1)
                    if want < 0:
                        for j in range(i+1, len(stats)-1):
                            jline, jlevel = stats[j]
                            if jlevel >= 0:
                                if have == getlspace(lines[jline]):
                                    want = jlevel * 4
                                break
                    if want < 0:
                        for j in range(i-1, -1, -1):
                            jline, jlevel = stats[j]
                            if jlevel >= 0:
                                want = have + getlspace(after[jline-1]) - \
                                       getlspace(lines[jline])
                                break
                    if want < 0:
                        want = have
                else:
                    want = 0
            assert want >= 0
            have2want[have] = want
            diff = want - have
            if diff == 0 or have == 0:
                after.extend(lines[thisstmt:nextstmt])
            else:
                for line in lines[thisstmt:nextstmt]:
                    if diff > 0:
                        if line == "\n":
                            after.append(line)
                        else:
                            after.append(" " * diff + line)
                    else:
                        remove = min(getlspace(line), -diff)
                        after.append(line[remove:])
        return self.raw != self.after

    def write(self, f):
        f.writelines(self.after)

    # Line-getter for tokenize.
    def getline(self):
        if self.index >= len(self.lines):
            line = ""
        else:
            line = self.lines[self.index]
            self.index += 1
        return line

    # Line-eater for tokenize.
    def tokeneater(self, type, token, start, end, line):
        if type == tokenize.NEWLINE:
            self.find_stmt = 1
        elif type == tokenize.INDENT:
            self.find_stmt = 1
            self.level += 1
        elif type == tokenize.DEDENT:
            self.find_stmt = 1
            self.level -= 1
        elif type == tokenize.COMMENT:
            if self.find_stmt:
                self.stats.append((start[0], -1))
        elif type == tokenize.NL:
            pass
        elif self.find_stmt:
            self.find_stmt = 0
            if line:
                self.stats.append((start[0], self.level))

# Count number of leading blanks.
def getlspace(line):
    i, n = 0, len(line)
    while i < n and line[i] == " ":
        i += 1
    return i

if __name__ == '__main__':
    main() 