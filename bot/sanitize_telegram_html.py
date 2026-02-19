from html.parser import HTMLParser

ALLOWED_TAGS = {
    'b',
    'strong',
    'i',
    'em',
    'u',
    'ins',
    's',
    'strike',
    'del',
    'tg-spoiler',
    'code',
    'pre',
    'blockquote',
    'a',
    'tg-emoji',
    'span',  # only allowed when class="tg-spoiler"
}


def _escape_text(s: str) -> str:
    # Telegram supports only &lt; &gt; &amp; &quot; as named entities
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


class TelegramHTMLSanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.out = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs = dict(attrs or [])

        if tag not in ALLOWED_TAGS:
            return  # drop tag, keep content via handle_data

        # filter attrs
        safe_attrs = ''

        if tag == 'a':
            href = attrs.get('href')
            if href:
                safe_attrs = f' href="{_escape_text(href)}"'
            else:
                return  # <a> without href not allowed/useful

        elif tag == 'span':
            # only spoiler span
            if attrs.get('class') != 'tg-spoiler':
                return
            safe_attrs = ' class="tg-spoiler"'

        elif tag == 'code':
            # allow class only for <pre><code class="language-...">
            cls = attrs.get('class')
            if cls:
                safe_attrs = f' class="{_escape_text(cls)}"'

        elif tag == 'blockquote':
            # allow expandable flag only
            if 'expandable' in attrs or any(a == 'expandable' for a in attrs.keys()):
                safe_attrs = ' expandable'

        elif tag == 'tg-emoji':
            eid = attrs.get('emoji-id')
            if eid:
                safe_attrs = f' emoji-id="{_escape_text(eid)}"'
            else:
                return

        # IMPORTANT: Telegram requires properly nested tags.
        self.out.append(f'<{tag}{safe_attrs}>')
        self.stack.append(tag)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if not self.stack:
            return

        # only close if it matches current open tag; otherwise ignore mismatched closes
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
            self.out.append(f'</{tag}>')

    def handle_data(self, data):
        self.out.append(_escape_text(data))

    def handle_entityref(self, name):
        # allow only supported named entities; convert others to plain text safely
        if name in ('lt', 'gt', 'amp', 'quot'):
            self.out.append(f'&{name};')
        else:
            # turn &nbsp; etc into text by escaping the &...;
            self.out.append(_escape_text(f'&{name};'))

    def handle_charref(self, name):
        # numeric entities allowed, keep them
        self.out.append(f'&#{name};')


def sanitize_telegram_html(s: str) -> str:
    p = TelegramHTMLSanitizer()
    p.feed(s)
    p.close()

    # close any still-open tags to keep it valid
    while p.stack:
        tag = p.stack.pop()
        p.out.append(f'</{tag}>')

    return ''.join(p.out)
