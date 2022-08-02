from pathlib import Path

class DisplayablePath(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))

def directory_files(dirct: Path) -> list:
    Lf = []
    if dirct.is_dir():
        Lf.append(dirct)
        for d in dirct.iterdir():
            Lf.extend(directory_files(d))
    else:
        Lf.append(dirct)   
    return Lf


if __name__ == '__main__':

    # https://docs.python.org/3/library/pathlib.html
    p = Path('.')

    # print([x for x in p.iterdir() if x.is_dir()])
    # print([x for x in p.iterdir()])
    
    # HIDDEN = ['__pycache__', '.ipynb_checkpoints', '.git']
    HIDDEN = []
    GIT = directory_files(Path('.git'))
    HIDDEN += GIT
    HIDDEN += directory_files(Path('./Data/chia_with_scope'))
    HIDDEN += directory_files(Path('./Data/chia_without_scope'))
    HIDDEN += list(p.glob('**/__pycache__'))
    HIDDEN += list(p.glob('**/.ipynb_checkpoints'))


    paths = DisplayablePath.make_tree(
        Path('.'),
        criteria=lambda path: True if path not in HIDDEN else False
        # criteria=is_not_hidden
    )
    for path in paths:
        print(path.displayable())



    # With a criteria (skip hidden files)
    # def is_not_hidden(path):
    #     return not path.name.startswith(".")

    # paths = DisplayablePath.make_tree(Path('.git'))
    # for path in paths:
    #     print(path.displayable())