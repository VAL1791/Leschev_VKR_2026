from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = ROOT / 'docs' / 'media'
SCREENS_DIR = MEDIA_DIR / 'screens'
TOUR_SRT = MEDIA_DIR / 'dashboard_tour.srt'
TOUR_MP4 = MEDIA_DIR / 'dashboard_tour.mp4'
MANIFEST = MEDIA_DIR / 'dashboard_tour_manifest.txt'
FFMPEG = Path(r'C:\Users\xsanf\Documents\ffmpeg\ffmpeg_folder\bin\ffmpeg.exe')
FONT_PATH = Path(r'C:\Windows\Fonts\arial.ttf')


@dataclass(frozen=True)
class TourClip:
    image_name: str
    duration_sec: float
    subtitle: str


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def make_title_card(target: Path, title: str, body_lines: Iterable[str]) -> None:
    image = Image.new('RGB', (1600, 900), color=(16, 22, 33))
    draw = ImageDraw.Draw(image)
    title_font = load_font(54)
    body_font = load_font(28)

    draw.rounded_rectangle((70, 70, 1530, 830), radius=28, fill=(25, 35, 52))
    draw.text((110, 120), title, font=title_font, fill=(245, 248, 252))

    y = 240
    for line in body_lines:
        draw.text((110, y), line, font=body_font, fill=(214, 222, 235))
        y += 54

    image.save(target)


def format_srt_timestamp(seconds: float) -> str:
    delta = timedelta(seconds=max(seconds, 0.0))
    total_ms = int(delta.total_seconds() * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}'


def build_assets() -> list[TourClip]:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)

    make_title_card(
        SCREENS_DIR / '00_intro.png',
        'Research Dashboard Tour',
        [
            'Короткий видео-тур по собранной ВКР-системе.',
            'Показываем, как пересобрать артефакты, как открыть dashboard',
            'и как читать основные страницы перед показом научруку.',
        ],
    )
    make_title_card(
        SCREENS_DIR / '00_run_commands.png',
        'How To Run',
        [
            '1. python -m src.run_real_benchmark --refresh',
            '2. streamlit run dashboard/app.py',
            '3. открыть Home и пройти страницы слева направо',
            '4. для защиты достаточно Data Audit, Benchmark, Robustness и Sandbox',
        ],
    )
    make_title_card(
        SCREENS_DIR / '07_closing.png',
        'What To Show',
        [
            'docs/RESULT_PRESENTATION_BRIEF.docx  — краткое пояснение результата',
            'docs/DEFENSE_SUMMARY.md             — ключевые выводы и интерпретация',
            'dashboard/artifacts/                — реальные benchmark-артефакты',
            'dashboard/app.py                    — точка входа в демонстрацию',
        ],
    )

    return [
        TourClip(
            image_name='00_intro.png',
            duration_sec=4.0,
            subtitle='Это короткий тур по собранной ВКР-системе: от запуска пайплайна до чтения финальных экранов dashboard.',
        ),
        TourClip(
            image_name='00_run_commands.png',
            duration_sec=5.5,
            subtitle='Сначала пересобираем реальные benchmark-артефакты, потом поднимаем Streamlit и показываем интерфейс как research dashboard, а не как магазин.',
        ),
        TourClip(
            image_name='01_home.png',
            duration_sec=6.0,
            subtitle='Home фиксирует карту проекта: workstreams, research questions, модельный scope и текущие takeaways. Это удобная точка входа для руководителя.',
        ),
        TourClip(
            image_name='02_data_audit_google_local.png',
            duration_sec=6.0,
            subtitle='Data Audit показывает, что данные не взялись из воздуха: здесь видны объёмы, sparsity, temporal activity и long-tail диагностика по выбранному датасету.',
        ),
        TourClip(
            image_name='03_model_benchmark_google_local.png',
            duration_sec=7.0,
            subtitle='Model Benchmark — главная страница сравнения: leaderboard, динамика по top-K, мульти-метрическое сравнение и статистическая надёжность результата.',
        ),
        TourClip(
            image_name='04_robustness_google_local.png',
            duration_sec=7.0,
            subtitle='Robustness and Business объясняет, за счёт чего модель сильна: inc versus exc, сегменты, feature importance, ablation и significance checks.',
        ),
        TourClip(
            image_name='05_sandbox_google_local.png',
            duration_sec=6.5,
            subtitle='Recommendation Sandbox нужен для человеческого объяснения: можно выбрать пользователя, посмотреть историю и сравнить выдачу разных алгоритмов с причинами ранжирования.',
        ),
        TourClip(
            image_name='06_supervisor_brief.png',
            duration_sec=5.5,
            subtitle='Supervisor Brief собирает готовый защитный маршрут: как показывать систему, какие тезисы уже подтверждены и что считается финальным объёмом работ.',
        ),
        TourClip(
            image_name='07_closing.png',
            duration_sec=5.0,
            subtitle='Для представления результата достаточно brief в DOCX, defense summary, сам dashboard и реальные CSV-артефакты, из которых он питается.',
        ),
    ]


def write_srt(clips: list[TourClip]) -> None:
    start = 0.0
    lines: list[str] = []
    for index, clip in enumerate(clips, start=1):
        end = start + clip.duration_sec
        lines.extend(
            [
                str(index),
                f'{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}',
                clip.subtitle,
                '',
            ]
        )
        start = end
    TOUR_SRT.write_text('\n'.join(lines), encoding='utf-8')


def write_manifest(clips: list[TourClip]) -> None:
    manifest_lines: list[str] = []
    for clip in clips:
        relative = f"screens/{clip.image_name}".replace('\\', '/')
        manifest_lines.append(f"file '{relative}'")
        manifest_lines.append(f'duration {clip.duration_sec:.3f}')

    # ffmpeg concat demuxer expects the last file to be repeated so the final
    # duration is honored instead of being cut short on the last frame.
    manifest_lines.append(f"file 'screens/{clips[-1].image_name}'")
    MANIFEST.write_text('\n'.join(manifest_lines), encoding='utf-8')


def render_video() -> None:
    if not FFMPEG.exists():
        raise FileNotFoundError(f'ffmpeg not found: {FFMPEG}')

    subtitle_filter = (
        "subtitles=dashboard_tour.srt:"
        "force_style='FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,Alignment=2,MarginV=12'"
    )

    command = [
        str(FFMPEG),
        '-y',
        '-f',
        'concat',
        '-safe',
        '0',
        '-i',
        MANIFEST.name,
        '-vf',
        f'scale=1600:900,format=yuv420p,{subtitle_filter}',
        '-r',
        '30',
        TOUR_MP4.name,
    ]

    subprocess.run(command, cwd=MEDIA_DIR, check=True)


def main() -> None:
    clips = build_assets()
    missing = [clip.image_name for clip in clips if not (SCREENS_DIR / clip.image_name).exists()]
    if missing:
        raise FileNotFoundError(
            'Missing expected screen captures. Run `node tools/capture_dashboard_tour.js` first. '
            f'Missing: {", ".join(missing)}'
        )

    write_srt(clips)
    write_manifest(clips)
    render_video()
    print(f'Created {TOUR_MP4}')
    print(f'Created {TOUR_SRT}')


if __name__ == '__main__':
    main()
