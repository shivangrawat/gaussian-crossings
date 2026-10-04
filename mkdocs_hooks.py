"""MkDocs hooks for the documentation site (registered in mkdocs.yml)."""

import posixpath
import re

# Relative links to notebooks, such as href="02_custom_covariance.ipynb#section".
_NOTEBOOK_LINK = re.compile(
    r'href="(?![a-z][a-z0-9+.-]*:|/|#)([^"#]+\.ipynb)(#[^"]*)?"', flags=re.IGNORECASE
)


def on_page_content(html, page, config, files):
    """Point links between tutorial notebooks at their pages on the site.

    The notebooks link to each other by file name so that the links work in Jupyter and on
    GitHub. mkdocs-jupyter copies such links unchanged, and from a page served at
    ``tutorials/01_quickstart/`` they would resolve below that page instead of to the sibling
    tutorial.
    """
    if not page.file.src_uri.endswith(".ipynb"):
        return html
    folder = posixpath.dirname(page.file.src_uri)

    def rewrite(match):
        source = posixpath.normpath(posixpath.join(folder, match.group(1)))
        target = files.get_file_from_path(source)
        if target is None:
            return match.group(0)
        return f'href="{target.url_relative_to(page.file)}{match.group(2) or ""}"'

    return _NOTEBOOK_LINK.sub(rewrite, html)
