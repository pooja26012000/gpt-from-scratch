"""
Streams a Lichess monthly PGN dump using python-chess's PGN parser,
filters for human games within a target Elo range, drops games with
zero moves, and saves the first N matching games to a local PGN file.
"""
import chess.pgn
import zstandard as zstd
import io
import requests

LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2026-04.pgn.zst"
OUTPUT_PATH = "data/raw/games.pgn"
MIN_ELO = 1600
MAX_ELO = 2200
TARGET_GAMES = 15000


def stream_and_filter():
    response = requests.get(LICHESS_URL, stream=True)
    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(response.raw)
    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

    games_saved = 0
    exporter_kwargs = dict(headers=True, variations=False, comments=False)

    with open(OUTPUT_PATH, "w") as out_file:
        while games_saved < TARGET_GAMES:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                print("Reached end of source file before hitting target.")
                break

            headers = game.headers
            if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
                continue

            try:
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
            except ValueError:
                continue

            if not (MIN_ELO <= white_elo <= MAX_ELO and MIN_ELO <= black_elo <= MAX_ELO):
                continue

            move_count = sum(1 for _ in game.mainline_moves())
            if move_count == 0:
                continue

            exporter = chess.pgn.StringExporter(**exporter_kwargs)
            out_file.write(game.accept(exporter) + "\n\n")
            games_saved += 1

            if games_saved % 500 == 0:
                print(f"Saved {games_saved} games so far...")

    print(f"Done. Saved {games_saved} games total.")


if __name__ == "__main__":
    stream_and_filter()
