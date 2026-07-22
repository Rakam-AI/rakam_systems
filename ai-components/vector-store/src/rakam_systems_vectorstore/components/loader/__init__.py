# Lazy re-exports. The document loaders (adaptive/code/doc/eml/html/md/odt/pdf/
# tabular) pull heavy optional deps (docling, pymupdf, ...) from the `loaders`
# extra. Importing them eagerly here would force those deps onto anyone importing
# a sibling module — notably the standalone pgvector/neo4j ingestion loaders,
# which must import under only the light `ingestion` extra. So resolve each name
# on access instead of at package import.

_DOCUMENT_LOADERS = {
    "AdaptiveLoader": "adaptive_loader",
    "create_adaptive_loader": "adaptive_loader",
    "CodeLoader": "code_loader",
    "create_code_loader": "code_loader",
    "DocLoader": "doc_loader",
    "create_doc_loader": "doc_loader",
    "EmlLoader": "eml_loader",
    "create_eml_loader": "eml_loader",
    "HtmlLoader": "html_loader",
    "create_html_loader": "html_loader",
    "MdLoader": "md_loader",
    "create_md_loader": "md_loader",
    "OdtLoader": "odt_loader",
    "create_odt_loader": "odt_loader",
    "PdfLoaderLight": "pdf_loader_light",
    "create_pdf_loader_light": "pdf_loader_light",
    "TabularLoader": "tabular_loader",
    "create_tabular_loader": "tabular_loader",
}

_INGESTION_LOADERS = {
    "PgVectorLoader": "pgvector_loader",
    "PgVectorLoaderConfig": "pgvector_loader",
}


def __getattr__(name):
    module = _DOCUMENT_LOADERS.get(name) or _INGESTION_LOADERS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f".{module}", __name__)
    return getattr(mod, name)


__all__ = [
    *_DOCUMENT_LOADERS,
    *_INGESTION_LOADERS,
]
