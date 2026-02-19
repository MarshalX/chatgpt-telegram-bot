import html
from html.parser import HTMLParser


class _HTMLToText(HTMLParser):
    BLOCK_START = {'p', 'div', 'pre', 'ul', 'ol', 'blockquote'}
    BLOCK_END = {'p', 'div', 'pre', 'ul', 'ol', 'blockquote', 'hr'}
    LINE_BREAK = {'br'}
    LIST_ITEM = {'li'}

    def __init__(self):
        super().__init__()
        self.out = []
        self._in_pre = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.LINE_BREAK:
            self.out.append('\n')
        elif tag in self.BLOCK_START:
            self.out.append('\n')
        elif tag in self.LIST_ITEM:
            self.out.append('\n- ')
        elif tag == 'pre':
            self._in_pre = True

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.BLOCK_END:
            self.out.append('\n')
        if tag == 'pre':
            self._in_pre = False

    def handle_data(self, data):
        self.out.append(data)

    def handle_entityref(self, name):
        self.out.append(f'&{name};')

    def handle_charref(self, name):
        self.out.append(f'&#{name};')


def html_to_text_pretty(s: str) -> str:
    p = _HTMLToText()
    p.feed(s)
    p.close()
    text = html.unescape(''.join(p.out))

    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == '':
            blank_run += 1
            if blank_run <= 2:
                cleaned.append('')
        else:
            blank_run = 0
            cleaned.append(ln)
    return '\n'.join(cleaned).strip()
