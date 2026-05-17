"""Виправляє типові помилки YouTube auto-ASR в українському тексті.

Стратегія: КОНСЕРВАТИВНА — тільки безпечні заміни на основі словника.
Не намагаємось виправляти hapax або робити загальну корекцію, бо це може
зіпсувати легітимні рідкісні слова.
"""
import re
import sys
from pathlib import Path

# 1) Помилки розпізнавання прізвища "Портников" (Vitaly's surname)
#    ВАЖЛИВО: НЕ чіпаємо "Португалі*", "Порту" (Porto), "Порті" (порт),
#    "Портнов" (інше прізвище — може бути політик Андрій Портнов).
#    Тому міняємо тільки коли впевнені.
PORTNIKOV_VARIANTS = [
    "Портиков", "Портнеков", "Портнаков", "Портаков",
    "Портнеко", "Портнев", "Портнікова", "Портніков",
    "Портнов", "Портнова", "Портновим",
    "Портков", "Портник",  # user-confirmed ASR errors
]

# Шумові маркери з ASR — видаляємо разом з оточуючими пробілами
NOISE_MARKERS = [
    "[музика]", "[оплески]", "[сопіння]",
    "[аплодисменти]", "[сміх]", "[Музика]", "[Оплески]",
]

# 2) Власні імена — auto-ASR часто опускає велику літеру в косвених відмінках.
#    Тільки ІМЕННИКИ-власні назви. Прикметники (українські, путінського,
#    трампістів) в українській НОРМАТИВНО з малої літери — не чіпаємо.
PROPER_NOUNS = {
    # Українські політики (тільки сам прізвище у відмінках)
    "зеленський", "зеленського", "зеленському", "зеленським", "зеленському",
    # Російські
    "путін", "путіна", "путіну", "путіним", "путіном",
    # Американські
    "трамп", "трампа", "трампу", "трампом", "трампі",
    "байден", "байдена", "байдену", "байденом", "байдені",
    # Європейські
    "орбан", "орбана", "орбану", "орбаном",
    "мерц", "мерца", "мерцу", "мерцем",
    "макрон", "макрона", "макрону", "макроном",
    # Країни — ТІЛЬКИ форми іменника (не прикметники!)
    "україна", "україни", "україні", "україну", "україною",
    "росія", "росії", "росію", "росією", "росією",
    "європа", "європи", "європі", "європу", "європою",
    # Міста
    "москва", "москви", "москві", "москву", "москвою",
    "київ", "києва", "києву", "києві", "києвом",
    "вашингтон", "вашингтона", "вашингтону",
    "берлін", "берліна", "берліну",
    "брюссель", "брюсселя", "брюсселю",
    # Регіони — proper nouns
    "донбас", "донбасу", "донбасі",
    "крим", "криму", "кримом",
}

# 3) Окремі очевидні artifacts (одиничні випадки)
EXACT_FIXES = {
    "байтен": "Байден",
    "морбаном": "Орбаном",
    "Украіна": "Україна",
    "Украіни": "України",
    "Украіні": "Україні",
    "украіна": "Україна",
    "украіни": "України",
    "украіні": "Україні",
}


def fix_text(text: str) -> tuple[str, dict[str, int]]:
    stats: dict[str, int] = {}

    # Pass 0: видалення шумових ASR маркерів
    for marker in NOISE_MARKERS:
        n = text.count(marker)
        if n:
            text = text.replace(marker, "")
            stats[f"noise:{marker}"] = n
    # Прибираємо подвійні пробіли після видалення маркерів
    text = re.sub(r"  +", " ", text)
    # Прибираємо пробіли на початку рядків
    text = re.sub(r"\n +", "\n", text)

    # Pass 1: точні заміни artifacts
    for bad, good in EXACT_FIXES.items():
        rx = re.compile(rf"\b{re.escape(bad)}\b")
        n = len(rx.findall(text))
        if n:
            text = rx.sub(good, text)
            stats[f"exact:{bad}→{good}"] = n

    # Pass 2: Портников variants → Портников (з відповідним відмінком)
    # Беремо тільки nominative форму — для більшої точності
    for variant in PORTNIKOV_VARIANTS:
        rx = re.compile(rf"\b{variant}\b")
        n = len(rx.findall(text))
        if n:
            # зберігаємо capitalization
            text = rx.sub("Портников", text)
            stats[f"portnikov:{variant}"] = n

    # Pass 3: capitalization власних імен.
    # Заміняємо тільки коли слово в МАЛОМУ регістрі і збігається з відомим іменем,
    # АЛЕ НЕ на початку речення (там і так велика літера через інший механізм)
    # AND НЕ після крапки/знака запитання (також початок речення).
    def cap_replace(match: re.Match) -> str:
        word = match.group(0)
        if word.lower() in PROPER_NOUNS:
            return word[0].upper() + word[1:]
        return word

    # Знаходимо слова, які НЕ на початку речення
    # (тобто перед ними є хоч щось крім крапки/!/?/\n)
    def is_sentence_start(text: str, pos: int) -> bool:
        i = pos - 1
        while i >= 0 and text[i] in " \t":
            i -= 1
        if i < 0:
            return True
        return text[i] in ".!?\n"

    # Збираємо всі слова в нижньому регістрі, які треба капіталізувати
    pattern = re.compile(r"\b[а-яіїєґ\']+\b")
    chunks: list[str] = []
    last = 0
    cap_count: dict[str, int] = {}
    for m in pattern.finditer(text):
        word = m.group(0)
        if word in PROPER_NOUNS and not is_sentence_start(text, m.start()):
            chunks.append(text[last:m.start()])
            new_word = word[0].upper() + word[1:]
            chunks.append(new_word)
            last = m.end()
            cap_count[word] = cap_count.get(word, 0) + 1
    chunks.append(text[last:])
    text = "".join(chunks)
    for w, n in cap_count.items():
        stats[f"cap:{w}→{w[0].upper()+w[1:]}"] = n

    return text, stats


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    text = src.read_text(encoding="utf-8")
    print(f"Input:  {len(text):,} chars")
    fixed, stats = fix_text(text)
    print(f"Output: {len(fixed):,} chars")
    print(f"\nFixes applied (top 30):")
    for key, n in sorted(stats.items(), key=lambda x: -x[1])[:30]:
        print(f"  {n:5d}  {key}")
    total = sum(stats.values())
    print(f"\nTotal substitutions: {total:,}")
    dst.write_text(fixed, encoding="utf-8")
    print(f"Written → {dst}")


if __name__ == "__main__":
    main()
