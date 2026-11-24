"""Tests for watch_folders, the Inbox settle window and source-folder learning.

Scenario modelled: a third-party sorter (SaneBox) moves new Inbox mail into
@SaneLater/@SaneNews/@SaneCC shortly after delivery. The filter must

  * score and act on mail in the watched folders,
  * never learn from the sorter's own moves,
  * learn spam/ham from the user's moves between any source folder and Junk,
  * ignore unwatched folders such as @SaneBlackHole entirely,
  * hold off on Inbox mail until the sorter has had its turn.

Run: python -m pytest test_watch_folders.py
"""

import datetime as dt
import logging
import os
import sqlite3
import tempfile
import time

os.environ.setdefault("STATE_DIR", tempfile.mkdtemp(prefix="sf_test_"))

import pytest  # noqa: E402

import filter as f  # noqa: E402

LOG = logging.getLogger("test")
HDR = b"BODY[HEADER.FIELDS (MESSAGE-ID FROM SUBJECT)]"
SANE = ["@SaneLater", "@SaneNews", "@SaneCC"]


# ----- fake IMAP -----------------------------------------------------------


class FakeIMAP:
    def __init__(self, folders):
        self.f = {
            name: {"uv": 1000 + i, "next": 1, "msgs": {}}
            for i, name in enumerate(folders)
        }
        self.selected = None
        self.moves = []  # (src, dst, msgid)

    # mailbox helpers used by the tests
    def deliver(self, folder, msgid, age_s=0, seen=False, flags=()):
        raw = (
            f"Message-ID: <{msgid}>\r\nFrom: s@example.com\r\n"
            f"Subject: {msgid}\r\nTo: me@example.com\r\n\r\nbody of {msgid}\r\n"
        ).encode()
        fl = set(flags) | ({b"\\Seen"} if seen else set())
        when = dt.datetime.fromtimestamp(time.time() - age_s, tz=dt.timezone.utc)
        return self._put(folder, {"raw": raw, "flags": fl, "date": when, "msgid": msgid})

    def _put(self, folder, msg):
        box = self.f[folder]
        uid = box["next"]
        box["next"] += 1
        box["msgs"][uid] = dict(msg, flags=set(msg["flags"]))
        return uid

    def where(self, msgid):
        return [n for n, b in self.f.items() for m in b["msgs"].values() if m["msgid"] == msgid]

    def user_move(self, msgid, dst, flags=()):
        src = self.where(msgid)[0]
        uid = next(u for u, m in self.f[src]["msgs"].items() if m["msgid"] == msgid)
        msg = self.f[src]["msgs"].pop(uid)
        msg["flags"] |= set(flags)
        self._put(dst, msg)

    # IMAPClient surface
    def select_folder(self, folder, readonly=False):
        if folder not in self.f:
            raise f.IMAPClientError(f"no such folder {folder}")
        self.selected = folder
        return {b"UIDVALIDITY": self.f[folder]["uv"]}

    def _box(self):
        return self.f[self.selected]["msgs"]

    def search(self, crit):
        box = self._box()
        uids = sorted(box)
        if crit == ["ALL"]:
            return uids
        if crit[0] == "HEADER":
            return [u for u in uids if box[u]["msgid"] == crit[2]]
        if crit[0] == "BEFORE":
            return []
        unseen_only = crit[0] == "UNSEEN"
        rng = crit[-1]
        lo = int(rng.split(":")[0])
        out = [u for u in uids if u >= lo]
        if not out and uids:
            out = [max(uids)]  # IMAP "n:*" quirk
        if unseen_only:
            out = [u for u in out if b"\\Seen" not in box[u]["flags"]]
        return out

    def fetch(self, uids, items):
        box = self._box()
        out = {}
        for u in uids:
            if u not in box:
                continue
            m = box[u]
            d = {}
            for it in items:
                if it == b"BODY.PEEK[]":
                    d[b"BODY[]"] = m["raw"]
                elif it.startswith(b"BODY.PEEK[HEADER.FIELDS"):
                    d[it.replace(b"BODY.PEEK", b"BODY")] = m["raw"].split(b"\r\n\r\n")[0] + b"\r\n\r\n"
                elif it == b"FLAGS":
                    d[b"FLAGS"] = tuple(m["flags"])
                elif it == b"INTERNALDATE":
                    d[b"INTERNALDATE"] = m["date"]
            out[u] = d
        return out

    def add_flags(self, uid, flags):
        self._box()[uid]["flags"] |= set(flags)

    def move(self, uids, dst):
        for u in list(uids):
            msg = self._box().pop(u)
            self.moves.append((self.selected, dst, msg["msgid"]))
            self._put(dst, msg)


# ----- fixtures ------------------------------------------------------------


@pytest.fixture
def env(tmp_path, monkeypatch):
    db_path = tmp_path / "spamfilter.db"
    monkeypatch.setattr(f, "DB_PATH", db_path)
    monkeypatch.setattr(f, "STATE_DIR", tmp_path)
    f.init_db()
    db = f.Db("acct")
    scores = {}
    learned = []
    monkeypatch.setattr(
        f, "rspamd_scan",
        lambda raw, rcpt, mx, bayes_user=None: scores.get(f.parse_envelope(raw)[0], 0.0),
    )

    def _learn(raw, kind, user):
        learned.append((kind, f.parse_envelope(raw)[0]))
        return "learned"

    monkeypatch.setattr(f, "rspamd_learn", _learn)
    imap = FakeIMAP(["INBOX", "Junk", "Trash", "Junk/Train-Spam", "Junk/Trained-Spam",
                     "Junk/Train-Ham", "Junk/Trained-Ham", *SANE, "@SaneBlackHole"])
    yield db, imap, scores, learned
    db.close()


def mk_account(**over):
    base = dict(
        name="acct", user="me@example.com", password="x",
        imap_host="h", imap_port=993, ssl=True,
        inbox="INBOX", junk="Junk", trash="Trash",
        spam_train="Junk/Train-Spam", trained_spam="Junk/Trained-Spam",
        ham_train="Junk/Train-Ham", trained_ham="Junk/Trained-Ham",
        mode="move", threshold=8.0, min_threshold_allowed=5.0,
        reject_score_above=100.0,
        move_grace_seconds=0, learn_grace_seconds=0, idle_timeout=1500,
        poll_interval=600, junk_poll_interval=120,
        retention_check_interval=3600,
        max_moves_per_hour=30, max_learns_per_hour=50, max_train_per_run=100,
        safe_mode_unseen_cap=500,
        junk_retention_days=0, trained_retention_days=0,
        learn_from_moves=True, auto_special_folders=False,
        watch_folders=list(SANE), inbox_settle_seconds=900,
        new_mail_max_age_seconds=86400, watch_poll_interval=60,
    )
    base.update(over)
    acc = f.Account(**base)
    f.validate_account(acc)
    acc.folder_map = f.build_folder_map(acc, "/")
    acc.watch_resolved = list(acc.watch_folders)
    return acc


def cycle(db, imap, acc):
    """One pass of the account loop's mail-handling steps."""
    fm = acc.folder_map
    f.scan_inbox(imap, db, LOG, acc, fm)
    f.scan_watch_folders(imap, db, LOG, acc, fm)
    f.execute_due_moves(imap, db, LOG, acc, fm)
    f.poll_junk(imap, db, LOG, acc, fm)


def prime(db, imap, acc):
    """First pass initialises the per-folder bookmarks."""
    cycle(db, imap, acc)


# ----- tests ---------------------------------------------------------------


def test_preexisting_mail_in_watch_folders_is_ignored(env):
    db, imap, scores, learned = env
    imap.deliver("@SaneLater", "old-spam", age_s=60)
    scores["old-spam"] = 20.0
    acc = mk_account()
    prime(db, imap, acc)
    cycle(db, imap, acc)
    assert imap.where("old-spam") == ["@SaneLater"]
    assert learned == []


def test_spam_filed_by_sorter_is_moved_without_learning(env):
    db, imap, scores, learned = env
    acc = mk_account()
    prime(db, imap, acc)
    imap.deliver("INBOX", "spam1")
    scores["spam1"] = 15.0
    cycle(db, imap, acc)                      # unsettled in Inbox
    assert imap.where("spam1") == ["INBOX"]
    imap.user_move("spam1", "@SaneLater")     # SaneBox files it
    cycle(db, imap, acc)
    assert imap.where("spam1") == ["Junk"]
    assert ("@SaneLater", "Junk", "spam1") in imap.moves
    cycle(db, imap, acc)                      # Junk poll sees our own move
    assert learned == []


def test_ham_filed_by_sorter_stays_and_is_not_learned(env):
    db, imap, scores, learned = env
    acc = mk_account()
    prime(db, imap, acc)
    imap.deliver("INBOX", "news1")
    scores["news1"] = 1.0
    cycle(db, imap, acc)
    imap.user_move("news1", "@SaneNews")
    cycle(db, imap, acc)
    cycle(db, imap, acc)
    assert imap.where("news1") == ["@SaneNews"]
    assert learned == []
    assert db.get_message("news1")["our_score"] == 1.0


def test_inbox_mail_waits_for_settle_window(env):
    db, imap, scores, learned = env
    acc = mk_account()
    prime(db, imap, acc)
    imap.deliver("INBOX", "fresh", age_s=10)
    scores["fresh"] = 30.0
    cycle(db, imap, acc)
    assert imap.where("fresh") == ["INBOX"]
    assert db.get_message("fresh")["our_score"] is None
    # Time passes: SaneBox left it in the Inbox.
    box = imap.f["INBOX"]["msgs"]
    for m in box.values():
        m["date"] -= dt.timedelta(seconds=1000)
    cycle(db, imap, acc)
    assert imap.where("fresh") == ["Junk"]


def test_settled_mail_already_read_is_still_scored(env):
    db, imap, scores, learned = env
    acc = mk_account()
    prime(db, imap, acc)
    imap.deliver("INBOX", "read-early", age_s=10, seen=True)
    scores["read-early"] = 30.0
    cycle(db, imap, acc)
    for m in imap.f["INBOX"]["msgs"].values():
        m["date"] -= dt.timedelta(seconds=1000)
    cycle(db, imap, acc)
    assert imap.where("read-early") == ["Junk"]


def test_old_mail_moved_into_inbox_is_not_scored(env):
    db, imap, scores, learned = env
    acc = mk_account()
    prime(db, imap, acc)
    imap.deliver("INBOX", "archived", age_s=10 * 86400, seen=True)
    scores["archived"] = 30.0
    cycle(db, imap, acc)
    assert imap.where("archived") == ["INBOX"]


def test_user_move_inbox_to_junk_during_settle_learns_spam(env):
    db, imap, scores, learned = env
    acc = mk_account()
    prime(db, imap, acc)
    imap.deliver("INBOX", "phish", age_s=5)
    cycle(db, imap, acc)
    imap.user_move("phish", "Junk")
    cycle(db, imap, acc)                      # schedules pending spam
    cycle(db, imap, acc)                      # grace 0 -> learn
    assert ("spam", "phish") in learned


def test_user_move_from_sane_folder_to_junk_learns_spam(env):
    db, imap, scores, learned = env
    acc = mk_account()
    prime(db, imap, acc)
    imap.deliver("@SaneCC", "cc-spam")
    scores["cc-spam"] = 2.0
    cycle(db, imap, acc)
    assert imap.where("cc-spam") == ["@SaneCC"]
    imap.user_move("cc-spam", "Junk", flags=[b"$Junk"])
    cycle(db, imap, acc)
    assert ("spam", "cc-spam") in learned


def test_user_revert_from_junk_to_sane_folder_learns_ham(env):
    db, imap, scores, learned = env
    acc = mk_account()
    prime(db, imap, acc)
    imap.deliver("@SaneLater", "false-pos")
    scores["false-pos"] = 12.0
    cycle(db, imap, acc)
    assert imap.where("false-pos") == ["Junk"]
    imap.user_move("false-pos", "@SaneLater")
    cycle(db, imap, acc)                      # schedules pending ham
    cycle(db, imap, acc)
    assert ("ham", "false-pos") in learned
    assert imap.where("false-pos") == ["@SaneLater"]


def test_blackhole_is_never_scanned_or_learned(env):
    db, imap, scores, learned = env
    acc = mk_account()
    prime(db, imap, acc)
    imap.deliver("@SaneBlackHole", "bh1")
    scores["bh1"] = 50.0
    cycle(db, imap, acc)
    assert imap.where("bh1") == ["@SaneBlackHole"]
    assert db.get_message("bh1") is None
    imap.user_move("bh1", "Junk", flags=[b"$Junk"])
    cycle(db, imap, acc)
    cycle(db, imap, acc)
    assert learned == []


def test_user_moving_scored_mail_from_sane_to_inbox_is_not_rescored(env):
    db, imap, scores, learned = env
    acc = mk_account()
    prime(db, imap, acc)
    imap.deliver("@SaneLater", "wanted")
    scores["wanted"] = 3.0
    cycle(db, imap, acc)
    scores["wanted"] = 40.0                   # would now be spam if rescored
    imap.user_move("wanted", "INBOX")
    cycle(db, imap, acc)
    assert imap.where("wanted") == ["INBOX"]
    assert learned == []


def test_shadow_mode_changes_nothing(env):
    db, imap, scores, learned = env
    acc = mk_account(mode="shadow", junk_retention_days=10)
    prime(db, imap, acc)
    imap.deliver("@SaneLater", "s1")
    scores["s1"] = 50.0
    cycle(db, imap, acc)
    f.retention_sweep(imap, db, LOG, acc, acc.folder_map)
    assert imap.where("s1") == ["@SaneLater"]
    assert imap.moves == []
    assert db.get_message("s1")["our_action"] == "shadow"


def test_watch_folders_must_not_overlap_core_folders():
    with pytest.raises(SystemExit):
        mk_account(watch_folders=["Junk"])
    with pytest.raises(SystemExit):
        mk_account(watch_folders=["@SaneLater", "@SaneLater"])


def test_password_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_IMAP_PW", "s3cret")
    cfg = tmp_path / "accounts.yml"
    cfg.write_text(
        "accounts:\n  - name: a\n    imap_host: h\n    user: u@example.com\n"
        "    password_env: TEST_IMAP_PW\n    watch_folders: ['@SaneLater']\n"
    )
    (acc,) = f.load_accounts(cfg)
    assert acc.password == "s3cret"
    assert acc.watch_folders == ["@SaneLater"]


def test_pending_move_migration(tmp_path, monkeypatch):
    db_path = tmp_path / "old.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE messages (account TEXT, message_id TEXT, first_seen INTEGER,
              last_seen INTEGER, current_folder TEXT, moved_to_junk_at INTEGER,
              our_score REAL, our_action TEXT, learned_as TEXT, learned_at INTEGER,
              pending_learn TEXT, pending_learn_at INTEGER, sender TEXT, subject TEXT,
              received_at INTEGER, PRIMARY KEY (account, message_id));
            CREATE TABLE pending_move (account TEXT NOT NULL, uidvalidity INTEGER NOT NULL,
              uid INTEGER NOT NULL, message_id TEXT NOT NULL, flag_at INTEGER NOT NULL,
              PRIMARY KEY (account, uidvalidity, uid));
            INSERT INTO pending_move VALUES ('acct', 7, 3, 'm', 1);
            """
        )
    monkeypatch.setattr(f, "DB_PATH", db_path)
    monkeypatch.setattr(f, "STATE_DIR", tmp_path)
    f.init_db()
    db = f.Db("acct")
    rows = db.due_pending_moves("INBOX", 7, 0)
    assert [(r["uid"], r["message_id"]) for r in rows] == [(3, "m")]
    db.close()
