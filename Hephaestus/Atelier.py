# Hephaestus/Atelier.py — Elodin-style Retro (Orange) Atelier for Ardor (Dockable Sidebar)
# pip install PySide6
# Run:
#   python Atelier.py
#   python Atelier.py C:\\Users\\adm\\PycharmProjects\\ProjectArdor

from __future__ import annotations
import sys, time, os, subprocess, platform
from pathlib import Path

from PySide6.QtCore import (
    Qt, QRect, QSize, QTimer, QRegularExpression, QDir, QFileSystemWatcher,
    QDirIterator, QEvent,
)
from PySide6.QtGui import (
    QColor, QPainter, QFont, QTextFormat, QTextCharFormat,
    QSyntaxHighlighter, QAction, QTextDocument, QTextOption, QPalette,
    QKeySequence, QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTreeView, QFileSystemModel, QTabWidget, QPlainTextEdit,
    QLineEdit, QPushButton, QLabel, QStatusBar,
    QFrame, QFileDialog, QMessageBox, QListWidget, QAbstractItemView,
    QTextEdit, QDockWidget, QHeaderView, QListWidgetItem,
)

# ─────────────────────────────── Config ───────────────────────────────
DEFAULT_ARDOR_ROOT = Path(r"C:/Users/adm/PycharmProjects/ProjectArdor")

# ─────────────────────────────── Theme ────────────────────────────────
class Theme:
    """Orange retro-futuristic (Elodin-inspired) palette."""
    def __init__(self, *, dark: bool = True):
        if dark:
            self.bg     = QColor("#0a0907")
            self.panel  = QColor("#100e0b")
            self.panel2 = QColor("#13100b")
            self.fg     = QColor("#fff4e2")
            self.muted  = QColor("#d2b58d")
            self.accent = QColor("#ff9100")   # neon orange
            self.accent2= QColor("#ffc861")   # amber
            self.edge   = QColor("#3b2a1d")   # bronze stroke
            self.grid   = QColor("#2f2216")
            self.line   = QColor(255, 145, 0, 32)  # line highlight
            self.caret  = QColor("#1c140c")
        else:
            self.bg     = QColor("#fff8ed")
            self.panel  = QColor("#fff2e0")
            self.panel2 = QColor("#ffe9cc")
            self.fg     = QColor("#2b1a0d")
            self.muted  = QColor("#7a5a3c")
            self.accent = QColor("#d46b00")
            self.accent2= QColor("#ffb354")
            self.edge   = QColor("#e6cfb1")
            self.grid   = QColor("#f2d8b8")
            self.line   = QColor(212, 107, 0, 32)
            self.caret  = QColor("#f7e3cd")

# ───────────────────── Python Syntax Highlighter ──────────────────────
PY_KEYWORDS = set(
    "False None True and as assert async await break class continue def del elif else "
    "except finally for from global if import in is lambda nonlocal not or pass raise "
    "return try while with yield".split()
)

class PyHighlighter(QSyntaxHighlighter):
    def __init__(self, doc, theme: Theme):
        super().__init__(doc)
        self.th = theme
        self.rules: list[tuple[QRegularExpression, QTextCharFormat]] = []
        self._build()

    def fmt(self, color: str | QColor, bold=False, italic=False):
        f = QTextCharFormat()
        f.setForeground(QColor(color) if isinstance(color, str) else color)
        if bold: f.setFontWeight(QFont.DemiBold)
        if italic: f.setFontItalic(True)
        return f

    def _build(self):
        self.rules.clear()
        self.rules.append((QRegularExpression(r"#[^\n]*"), self.fmt("#d2b58d", italic=True)))
        str_color = self.fmt("#ffc861")
        self.rules.append((QRegularExpression(r'\"\"\"[\s\S]*?\"\"\"'), str_color))
        self.rules.append((QRegularExpression(r"'''[\s\S]*?'''"), str_color))
        self.rules.append((QRegularExpression(r'"(?:\\.|[^"\\\n])*"'), str_color))
        self.rules.append((QRegularExpression(r"'(?:\\.|[^'\\\n])*'"), str_color))
        self.rules.append((QRegularExpression(r"\b\d+(?:\.\d+)?\b"), self.fmt("#ffb86b")))
        self.rules.append((QRegularExpression(r"\b(import|from)\b[^\n]*"), self.fmt("#ffd18b")))
        name_fmt = self.fmt("#ff9100", bold=True)
        self.rules.append((QRegularExpression(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)"), name_fmt))
        self.kw_fmt = self.fmt("#ff7a1a", bold=True)
        self.kw_re = QRegularExpression(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")

    def highlightBlock(self, text: str):
        for rx, fmt in self.rules:
            it = rx.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)
        it = self.kw_re.globalMatch(text)
        while it.hasNext():
            m = it.next()
            if m.captured(1) in PY_KEYWORDS:
                self.setFormat(m.capturedStart(1), m.capturedLength(1), self.kw_fmt)

# ───────────────────────────── Code Editor ─────────────────────────────
class LineNumberArea(QWidget):
    def __init__(self, editor: 'CodeEditor'):
        super().__init__(editor)
        self.editor = editor
    def sizeHint(self):
        return QSize(self.editor.line_number_area_width(), 0)
    def paintEvent(self, event):
        self.editor.line_number_area_paint(event)

class CodeEditor(QPlainTextEdit):
    def __init__(self, theme: Theme):
        super().__init__()
        self.th = theme
        self.setFont(QFont("Consolas", 11))
        self.setFrameStyle(QFrame.NoFrame)
        self.setWordWrapMode(QTextOption.NoWrap)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(' '))
        self._lineArea = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.highlight_current_line)
        self.update_line_number_area_width(0)
        self.highlight_current_line()

        pal = self.palette()
        pal.setColor(QPalette.Base, self.th.panel)
        pal.setColor(QPalette.Text, self.th.fg)
        self.setPalette(pal)
        self.setStyleSheet(
            f"QPlainTextEdit {{ background: {self.th.panel.name()}; color: {self.th.fg.name()};"
            f" selection-background-color: rgba(255,145,0,.22); border:0; }}"
        )

    # line numbers
    def line_number_area_width(self):
        digits = len(str(max(1, self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance('9') * digits

    def update_line_number_area_width(self, _):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        if dy:
            self._lineArea.scroll(0, dy)
        else:
            self._lineArea.update(0, rect.y(), self._lineArea.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._lineArea.setGeometry(QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height()))

    def line_number_area_paint(self, event):
        painter = QPainter(self._lineArea)
        painter.fillRect(event.rect(), self.th.panel2)
        block = self.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        painter.setPen(QColor("#d1bfa6"))
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.drawText(0, top, self._lineArea.width()-6, self.fontMetrics().height(),
                                 Qt.AlignRight, str(blockNumber + 1))
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            blockNumber += 1

    def highlight_current_line(self):
        sel = []
        line_sel = QTextEdit.ExtraSelection()  # helper lives on QTextEdit
        fmt = QTextCharFormat()
        fmt.setBackground(self.th.line)
        fmt.setProperty(QTextFormat.FullWidthSelection, True)
        line_sel.format = fmt
        cur = self.textCursor()
        cur.clearSelection()
        line_sel.cursor = cur
        sel.append(line_sel)
        self.setExtraSelections(sel)

# ────────────────────────── Retro Scanline Bar ─────────────────────────
class Scanline(QFrame):
    def __init__(self, theme: Theme):
        super().__init__()
        self.th = theme
        self.setFixedHeight(3)
        self._x = -100
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(24)
        self.setStyleSheet("background: transparent; border:0;")
    def _tick(self):
        self._x = (self._x + 6) % (self.width() + 160)
        self.update()
    def paintEvent(self, _e):
        p = QPainter(self)
        x0 = self._x - 80
        p.fillRect(self.rect(), self.th.panel)
        p.fillRect(QRect(x0, 0, 80, self.height()), self.th.accent)
        p.fillRect(QRect(x0+80, 0, 16, self.height()), self.th.accent2)

# ─────────────────────────────── Main UI ──────────────────────────────
class AtelierQt(QMainWindow):
    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = Path(project_root).resolve()
        self.theme = Theme(dark=True)
        self.setWindowTitle("Hephaestus")
        self.resize(1280, 820)
        self._tabs: dict[int, dict] = {}
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._external_change)

        # central layout (no splitter; dock for sidebar)
        central = QWidget(); self.setCentralWidget(central)
        c_v = QVBoxLayout(central); c_v.setContentsMargins(12,12,12,12); c_v.setSpacing(8)

        # Top bar (brand + quick actions)
        hud = QFrame(); hud.setObjectName("hud")
        h = QHBoxLayout(hud); h.setContentsMargins(12,8,12,8); h.setSpacing(8)
        self.brand = QLabel("Atelier"); self.brand.setObjectName("brand")
        h.addWidget(self.brand); h.addStretch(1)
        self.btn_run = QPushButton("Run")
        self.btn_theme = QPushButton("Theme")
        self.btn_cortex = QPushButton("Cortex HUD")   # NEW
        for b in (self.btn_run, self.btn_theme, self.btn_cortex):
            b.setObjectName("pill"); b.setCursor(Qt.PointingHandCursor); h.addWidget(b)
        c_v.addWidget(hud)

        # find bar (SCAN)
        find = QHBoxLayout(); find.setSpacing(8)
        self.find_label = QLabel("SCAN:"); self.find_label.setObjectName("muted")
        self.find_edit = QLineEdit()
        self.btn_prev = QPushButton("Prev"); self.btn_next = QPushButton("Next")
        for b in (self.btn_prev, self.btn_next):
            b.setObjectName("chip"); b.setCursor(Qt.PointingHandCursor)
        find.addWidget(self.find_label); find.addWidget(self.find_edit, 1)
        find.addWidget(self.btn_prev); find.addWidget(self.btn_next)
        c_v.addLayout(find)

        # results list under SCAN
        self.scan_results = QListWidget()
        self.scan_results.setObjectName("scanlist")
        self.scan_results.setVisible(False)
        self.scan_results.setMaximumHeight(240)
        self.scan_results.setSelectionMode(QAbstractItemView.SingleSelection)
        self.scan_results.itemActivated.connect(self._open_scan_selection)
        c_v.addWidget(self.scan_results, 0)

        # debounce typing in SCAN bar
        self._scan_timer = QTimer(self)
        self._scan_timer.setSingleShot(True)
        self._scan_timer.timeout.connect(self._run_scan)
        self.find_edit.textChanged.connect(lambda: self._scan_timer.start(220))

        # Keys: Enter opens top hit; Down focuses list; Esc hides results
        self.find_edit.returnPressed.connect(
            lambda: self._open_scan_selection(self.scan_results.item(0)) if self.scan_results.count() else None
        )
        self.find_edit.installEventFilter(self)
        self.scan_results.installEventFilter(self)

        # scanline
        self.scan = Scanline(self.theme); c_v.addWidget(self.scan)

        # CENTER: tabs + editor
        center = QFrame(); center.setObjectName("pane")
        cv = QVBoxLayout(center); cv.setContentsMargins(8,8,8,8); cv.setSpacing(6)
        self.tabs = QTabWidget(); self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        cv.addWidget(self.tabs, 1)
        c_v.addWidget(center, 1)

        # Dockable LEFT SIDEBAR (movable/floating)
        self.projectDock = QDockWidget(f"Project — {self.project_root}", self)
        self.projectDock.setObjectName("projectDock")
        self.projectDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.projectDock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        dockw = QWidget(); dv = QVBoxLayout(dockw); dv.setContentsMargins(6,6,6,6); dv.setSpacing(6)
        # File tree
        self.model = QFileSystemModel(); self.model.setRootPath(str(self.project_root))
        self.model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)
        self.tree = QTreeView(); self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(str(self.project_root)))
        self.tree.doubleClicked.connect(self._tree_open)
        self.tree.setObjectName("tree")
        self.tree.setHeaderHidden(False)
        self.tree.setIndentation(18); self.tree.setAnimated(True)
        self.tree.setIconSize(QSize(16,16)); self.tree.setTextElideMode(Qt.ElideMiddle)
        hdr = self.tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, self.model.columnCount()):
            self.tree.setColumnHidden(c, True)
        dv.addWidget(self.tree, 1)
        self.projectDock.setWidget(dockw)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.projectDock)

        # (transparent overlay holder for consistency)
        self._crt = QFrame(self)
        self._crt.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._crt.setStyleSheet("background: transparent;")
        QTimer.singleShot(0, lambda: self._crt.setGeometry(self.rect()))

        # status bar + styling
        sb = QStatusBar(); self.setStatusBar(sb)
        self._style_window()

        # actions / shortcuts
        self._make_actions()

        # wire header buttons
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_prev.clicked.connect(lambda: self.find_next(False))
        self.btn_next.clicked.connect(lambda: self.find_next(True))
        self.btn_cortex.clicked.connect(self.switch_to_cortex)   # NEW

        # (optional) simple runner: run current file with Python
        self.btn_run.clicked.connect(self._run_current_file)

    # ─────────────────────────── Styling ────────────────────────────
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._crt: self._crt.setGeometry(self.rect())

    def _style_window(self):
        th = self.theme
        sel_col = "#ffffff" if (th.bg == QColor("#0a0907")) else "#000000"

        base = (
            f"QWidget {{ background: {th.bg.name()}; color: {th.fg.name()}; }}"
            f"QFrame#hud {{ border-bottom:1px solid {th.edge.name()}; "
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {th.panel.name()}, stop:1 {th.panel2.name()}); }}"
            f"QLabel#brand {{ color:{th.fg.name()}; font-weight:600; letter-spacing:2px; }}"
            f"QLabel#muted {{ color:{th.muted.name()}; }}"
            f"QFrame#pane {{ border:1px solid {th.edge.name()}; border-radius:18px; "
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {th.panel.name()}, stop:1 {th.panel2.name()}); }}"
            f"QDockWidget#projectDock {{ border:1px solid {th.edge.name()}; border-radius:12px; }}"
            f"QDockWidget::title {{ text-transform:uppercase; letter-spacing:2px; color:{th.muted.name()}; padding:6px 8px; }}"
            f"QTreeView#tree {{ background:{th.panel.name()}; color:{th.fg.name()}; border:1px solid {th.edge.name()}; border-radius:10px; }}"
            f"QTreeView#tree::item:selected {{ background: rgba(255,145,0,.16); color:{th.fg.name()}; }}"
            f"QTabBar::tab {{ background:{th.panel.name()}; padding:6px 10px; border:1px solid {th.edge.name()}; "
            f"border-bottom: none; border-top-left-radius:12px; border-top-right-radius:12px; color:{th.fg.name()}; }}"
            f"QTabBar::tab:selected {{ color:{sel_col}; border-color:#5a3a1f; }}"
            f"QTabWidget::pane {{ border:1px solid {th.edge.name()}; top:-1px; }}"
            f"QLineEdit {{ background: {th.panel2.name()}; border:1px solid {th.edge.name()}; padding:6px; }}"
            f"QStatusBar {{ background:{th.panel.name()}; border-top:1px solid {th.edge.name()}; }}"
            f"QListWidget#scanlist {{ background:{th.panel2.name()}; border:1px solid {th.edge.name()};"
            f" border-radius:10px; color:{th.fg.name()}; }}"
            f"QListWidget#scanlist::item:selected {{ background: rgba(255,145,0,.16); }}"
        )
        chips = (
            "QPushButton#chip {"
            f"  color:{th.fg.name()}; border:1px solid {th.edge.name()}; padding:6px 10px; border-radius:10px;"
            f"  background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {th.panel.name()}, stop:1 {th.panel2.name()});"
            "}"
            "QPushButton#chip:hover {"
            f"  border-color:{th.accent.name()};"
            "}"
            "QPushButton#pill {"
            f"  color:{th.fg.name()}; border:1px solid {th.edge.name()}; padding:8px 12px; border-radius:999px;"
            f"  background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {th.panel.name()}, stop:1 {th.panel2.name()});"
            "}"
            "QPushButton#pill:hover {"
            f"  border-color:{th.accent.name()};"
            "}"
        )
        self.setStyleSheet(base + chips)

    # ─────────────────────────── Actions ─────────────────────────────
    def _make_actions(self):
        mk = lambda text, seq, handler: QAction(text, self, shortcut=seq, triggered=handler)
        actions = [
            mk("New File", QKeySequence.New, self.new_file),
            mk("Open…", QKeySequence.Open, self.browse_open),
            mk("Save", QKeySequence.Save, self.save_active),
            mk("Save As…", QKeySequence("Ctrl+Shift+S"), self.save_as),
            mk("Close Tab", QKeySequence.Close, lambda: self._close_tab(self.tabs.currentIndex())),
            mk("Find", QKeySequence.Find, lambda: self.find_edit.setFocus()),
            mk("Find Next", QKeySequence.FindNext, lambda: self.find_next(True)),
            mk("Find Prev", QKeySequence.FindPrevious, lambda: self.find_next(False)),
            mk("Toggle Theme", QKeySequence("Ctrl+T"), self.toggle_theme),
            mk("Quick Search", QKeySequence("Ctrl+P"),
               lambda: (self.find_edit.setFocus(), self.find_edit.selectAll())),
        ]
        for a in actions: self.addAction(a)

    # ───────────────────────────── Tree ──────────────────────────────
    def _tree_open(self, idx):
        path = Path(self.model.filePath(idx))
        if path.is_file():
            self.open_file(path)

    def refresh_tree(self):
        try:
            self.model.setRootPath(self.model.rootPath())
        except Exception:
            pass
        self.statusBar().showMessage("Tree refreshed", 1200)

    # ───────────────────────────── Tabs ──────────────────────────────
    def editor_by_index(self, i: int) -> QPlainTextEdit | None:
        w = self.tabs.widget(i)
        return getattr(w, 'editor', None)

    def _mk_editor_tab(self, path: Path, content: str) -> int:
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0,0,0,0)
        editor = CodeEditor(self.theme)
        editor.setPlainText(content)
        PyHighlighter(editor.document(), self.theme)
        v.addWidget(editor)
        container.editor = editor  # type: ignore
        idx = self.tabs.addTab(container, path.name)
        self._tabs[idx] = {'path': path, 'editor': editor, 'dirty': False}
        editor.textChanged.connect(lambda: self._mark_dirty(idx, True))
        self.tabs.setCurrentIndex(idx)
        self.statusBar().showMessage(f"Opened {path}", 1500)
        try:
            if path.exists():
                self._watcher.addPath(str(path))
        except Exception:
            pass
        return idx

    def open_file(self, path: Path):
        # focus existing tab if open
        for i, meta in list(self._tabs.items()):
            if meta['path'] == path:
                self.tabs.setCurrentIndex(i)
                return
        try:
            content = path.read_text(encoding='utf-8')
        except Exception:
            try:
                content = path.read_text(errors='ignore')
            except Exception as e:
                QMessageBox.critical(self, "Open failed", str(e))
                return
        idx = self._mk_editor_tab(path, content)
        self._focus_editor_on_scan_match(idx)

    def _close_tab(self, index: int):
        meta = self._tabs.get(index)
        if not meta:
            return
        if meta['dirty']:
            r = QMessageBox.question(self, "Unsaved", f"Save changes to {meta['path'].name}?")
            if r == QMessageBox.Yes:
                self.save_index(index)
        try:
            self._watcher.removePath(str(meta['path']))
        except Exception:
            pass
        self.tabs.removeTab(index)
        self._tabs.pop(index, None)

    def _mark_dirty(self, index: int, dirty: bool):
        meta = self._tabs.get(index)
        if not meta or meta['dirty'] == dirty:
            return
        meta['dirty'] = dirty
        self.tabs.setTabText(index, meta['path'].name + (" *" if dirty else ""))

    def _external_change(self, changed_path: str):
        for i, meta in self._tabs.items():
            if str(meta['path']) == changed_path:
                if QMessageBox.question(self, "File changed on disk",
                                        f"Reload changes from disk?\n{changed_path}") == QMessageBox.Yes:
                    try:
                        txt = Path(changed_path).read_text(encoding='utf-8')
                    except Exception:
                        txt = Path(changed_path).read_text(errors='ignore')
                    meta['editor'].setPlainText(txt)
                    self._mark_dirty(i, False)
                break

    # ─────────────────────────── File ops ────────────────────────────
    def _current_index(self) -> int:
        return self.tabs.currentIndex()

    def _current_meta(self):
        return self._tabs.get(self._current_index())

    def new_file(self):
        p, _ = QFileDialog.getSaveFileName(self, "New file", str(self.project_root), "All (*.*)")
        if not p:
            return
        try:
            Path(p).write_text("", encoding='utf-8')
        except Exception as e:
            QMessageBox.critical(self, "Create failed", str(e))
            return
        self.open_file(Path(p))

    def browse_open(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open", str(self.project_root), "All (*.*)")
        if p:
            self.open_file(Path(p))

    def save_index(self, index: int):
        meta = self._tabs.get(index)
        if not meta:
            return
        text = meta['editor'].toPlainText()
        try:
            meta['path'].write_text(text, encoding='utf-8')
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        self._mark_dirty(index, False)
        self.statusBar().showMessage(f"Saved: {meta['path']}", 1500)

    def save_active(self):
        i = self._current_index()
        if i >= 0:
            self.save_index(i)

    def save_as(self):
        meta = self._current_meta()
        if not meta:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save As", str(meta['path']))
        if p:
            meta['path'] = Path(p)
            self.save_active()
            self.tabs.setTabText(self._current_index(), meta['path'].name)
            try:
                self._watcher.addPath(str(meta['path']))
            except Exception:
                pass

    def delete_active(self):
        meta = self._current_meta()
        if not meta:
            return
        if QMessageBox.question(self, "Delete", f"Delete {meta['path']}?") == QMessageBox.Yes:
            try:
                meta['path'].unlink(missing_ok=True)
                self._close_tab(self._current_index())
            except Exception as e:
                QMessageBox.critical(self, "Delete failed", str(e))

    # ───────────────────────────── Find / SCAN ─────────────────────────
    def find_next(self, forward=True):
        meta = self._current_meta()
        if not meta:
            return
        editor: CodeEditor = meta['editor']
        flags = QTextDocument.FindFlags()
        if not forward:
            flags |= QTextDocument.FindBackward
        pattern = self.find_edit.text()
        if not pattern:
            return
        cursor = editor.textCursor()
        if not editor.find(pattern, flags):
            # wrap
            cursor.movePosition(QTextCursor.Start if forward else QTextCursor.End)
            editor.setTextCursor(cursor)
            editor.find(pattern, flags)

    def _run_scan(self):
        q = self.find_edit.text().strip()
        self.scan_results.clear()
        if not q:
            self.scan_results.setVisible(False)
            return

        # File-name search (ordered token contains)
        name_hits = []
        it = QDirIterator(str(self.project_root), QDir.Files, QDirIterator.Subdirectories)
        toks = q.lower().split()

        def ok_name(s: str):
            i = 0
            for t in toks:
                j = s.find(t, i)
                if j == -1:
                    return False
                i = j + len(t)
            return True

        while it.hasNext() and len(name_hits) < 200:
            p = Path(it.next())
            rel = str(p.relative_to(self.project_root)).lower()
            if ok_name(rel):
                name_hits.append(p)

        # Content search
        content_hits = []
        max_files = 200
        it2 = QDirIterator(str(self.project_root), QDir.Files, QDirIterator.Subdirectories)
        q_lower = q.lower()
        while it2.hasNext() and len(content_hits) < max_files:
            p = Path(it2.next())
            try:
                if p.stat().st_size > 2_000_000:  # 2MB skip
                    continue
                text = p.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            idx = text.lower().find(q_lower)
            if idx != -1:
                line_no = text[:idx].count("\n") + 1
                start = max(0, idx - 40)
                end = min(len(text), idx + 60)
                snippet = text[start:end].replace("\n", " ")
                content_hits.append((p, line_no, snippet))

        # Populate list
        def add_item(text, data):
            it = QListWidgetItem(text)
            it.setData(Qt.UserRole, data)
            self.scan_results.addItem(it)

        for p in name_hits[:200]:
            add_item(f"[file] {p.relative_to(self.project_root)}", {"path": str(p), "type": "file"})

        for p, line, snippet in content_hits[:200]:
            add_item(f"[hit]  {p.relative_to(self.project_root)}  —  L{line}: {snippet}",
                     {"path": str(p), "type": "content", "line": line, "q": q})

        self.scan_results.setVisible(self.scan_results.count() > 0)

    def _open_scan_selection(self, item: QListWidgetItem | None):
        if not item:
            return
        info = item.data(Qt.UserRole) or {}
        path = Path(info.get("path", ""))
        if not path.exists():
            return
        self.open_file(path)
        if info.get("type") == "content":
            self.find_edit.setText(info.get("q", ""))
            self._focus_editor_on_scan_match(self._current_index())

    def _focus_editor_on_scan_match(self, idx: int):
        meta = self._tabs.get(idx)
        if not meta:
            return
        editor: CodeEditor = meta['editor']
        cursor = editor.textCursor()
        cursor.movePosition(QTextCursor.Start)
        editor.setTextCursor(cursor)
        pattern = self.find_edit.text().strip()
        if pattern:
            editor.find(pattern, QTextDocument.FindFlags())

    # Let Down arrow move focus into results; Esc hides list
    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.KeyPress:
            if obj == self.find_edit:
                if ev.key() == Qt.Key_Down and self.scan_results.isVisible() and self.scan_results.count():
                    self.scan_results.setFocus()
                    self.scan_results.setCurrentRow(0)
                    return True
                if ev.key() == Qt.Key_Escape:
                    self.scan_results.setVisible(False)
                    return True
            if obj == self.scan_results:
                if ev.key() == Qt.Key_Escape:
                    self.find_edit.setFocus()
                    self.scan_results.setVisible(False)
                    return True
        return super().eventFilter(obj, ev)

    # ───────────────────────────── Theme ──────────────────────────────
    def toggle_theme(self):
        is_dark = (self.theme.bg == QColor("#0a0907"))
        self.theme = Theme(dark=not is_dark)
        self._style_window()
        # re-style editors + rebind highlighter with new theme
        for _, meta in self._tabs.items():
            e: CodeEditor = meta['editor']
            e.th = self.theme
            PyHighlighter(e.document(), self.theme)
            pal = e.palette()
            pal.setColor(QPalette.Base, self.theme.panel)
            pal.setColor(QPalette.Text, self.theme.fg)
            e.setPalette(pal)
            e.setStyleSheet(
                f"QPlainTextEdit {{ background: {self.theme.panel.name()}; color: {self.theme.fg.name()};"
                f" selection-background-color: rgba(255,145,0,.22); border:0; }}"
            )
        self.statusBar().showMessage("Theme toggled", 1200)

    # ───────────────────────────── Runner / Switch ─────────────────────
    def _run_current_file(self):
        meta = self._current_meta()
        if not meta:
            self.statusBar().showMessage("No file selected", 1500)
            return
        self.save_active()
        script = meta['path']
        if script.suffix.lower() != ".py":
            QMessageBox.information(self, "Run", "Select a .py file to run.")
            return
        env = os.environ.copy()
        env.setdefault("ARDOR_HOME", str(self.project_root))
        kwargs = {}
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            subprocess.Popen([sys.executable, str(script)], cwd=str(script.parent), env=env, **kwargs)
            self.statusBar().showMessage(f"Running {script.name}…", 2000)
        except Exception as e:
            QMessageBox.critical(self, "Run failed", str(e))

    def _find_cortex_script(self) -> Path | None:
        # Common likely locations relative to this file and project root
        candidates: list[Path] = [
            self.project_root / "GUI_Cortex.py",
            Path(__file__).resolve().parent.parent / "GUI_Cortex.py",  # project_root/.. from Hephaestus
            Path.cwd() / "../Praetor/GUI_Cortex.py",
        ]
        for p in candidates:
            if p.exists():
                return p.resolve()
            else:
                print(f"{p} is not in canditdates!")
        return None

    def switch_to_cortex(self):
        # Optionally offer to save all modified tabs
        dirty_tabs = [i for i, m in self._tabs.items() if m.get("dirty")]
        if dirty_tabs:
            if QMessageBox.question(self, "Save changes",
                                    "Save all modified files before switching to Cortex HUD?") == QMessageBox.Yes:
                for i in list(self._tabs.keys()):
                    self.save_index(i)

        cortex = self._find_cortex_script()
        if cortex is None:
            QMessageBox.critical(self, "Not found", "Could not locate GUI_Cortex.py")
            return

        env = os.environ.copy()
        env.setdefault("ARDOR_HOME", str(self.project_root))
        kwargs = {}
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            subprocess.Popen([sys.executable, str(cortex)], cwd=str(cortex.parent), env=env, **kwargs)
        except Exception as e:
            QMessageBox.critical(self, "Launch failed", str(e))
            return

        # Close this window/app after launching Cortex
        self.close()
        QTimer.singleShot(200, QApplication.instance().quit)

# ──────────────────────────────── Entrypoints ───────────────────────────────

def _resolve_root_from_argv() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            return p.resolve()
    if DEFAULT_ARDOR_ROOT.exists():
        return DEFAULT_ARDOR_ROOT.resolve()
    return Path('.').resolve()

def main():
    app = QApplication(sys.argv)
    root = _resolve_root_from_argv()
    w = AtelierQt(root)
    w.show()
    sys.exit(app.exec())

# Convenience launcher used by Tk GUI_Cortex.py:
# GUI side calls:  Atelier(self, ROOT_DIR)
def Atelier(_parent=None, root_dir: str | Path | None = None):
    """Spawn a new process running this Atelier UI (for use from Tk)."""
    try:
        here = Path(__file__).resolve()
    except Exception:
        here = Path("Atelier.py")
    script = str(here)
    root = Path(root_dir) if root_dir else _resolve_root_from_argv()
    env = os.environ.copy()
    env.setdefault("ARDOR_HOME", str(root.resolve()))
    kwargs = {}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([sys.executable, script, str(root)], cwd=str(here.parent), env=env, **kwargs)

if __name__ == '__main__':
    main()
