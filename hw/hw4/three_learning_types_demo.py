"""
Урок 4: ML in Production — Три типи навчання
Чим відрізняються і де використовуються.

1. Supervised learning   — є правильні відповіді (fraud detection, spam, credit scoring)
2. Unsupervised learning — немає відповідей, шукаємо структуру (кластери, anomalies)
3. Reinforcement learning — агент вчиться через нагороду/штраф (RLHF, ігри, роботи)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

# ══════════════════════════════════════════════════════════════
# 1. SUPERVISED LEARNING
# ══════════════════════════════════════════════════════════════
# З правильною відповіддю на кожен приклад.
# Модель вчиться на парах: вхід → правильна відповідь.
# Як учень з підручником де є ключі до задач.
#
# Приклади: fraud detection, spam filter, credit scoring, fine-tuning LLM

print("=" * 65)
print("1. SUPERVISED LEARNING — є правильні відповіді")
print("=" * 65)
print()

# Дані: email → spam чи ні
emails = pd.DataFrame({
    "довжина":          [12, 150, 8, 200, 15, 180, 10, 160, 20, 190],
    "кількість_посилань": [5, 0, 8, 1, 6, 0, 7, 0, 4, 1],
    "великі_літери_%":  [40, 5, 60, 8, 35, 3, 55, 6, 30, 7],
    "spam":             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # ← МІТКИ є
})

print("Задача: spam detection")
print("Дані: кожен email має мітку — spam (1) чи ні (0)")
print(emails.to_string(index=False))
print()

# X = фічі (3 колонки), y = мітки (1 = spam, 0 = ні)
# LogisticRegression попри назву — це КЛАСИФІКАТОР, не regression.
# Під капотом: score = w1*фіча1 + w2*фіча2 + w3*фіча3 + bias
# Потім sigmoid стискає score в ймовірність 0-1, поріг 0.5 → клас.
X = emails[["довжина", "кількість_посилань", "великі_літери_%"]]
y = emails["spam"]

model = LogisticRegression(random_state=42)
# fit() підбирає ваги (w1, w2, w3, bias) через gradient descent —
# крок за кроком зсуває ваги в напрямку зменшення помилки.
model.fit(X, y)
# predict() рахує score з натренованими вагами і повертає клас.
# УВАГА: predict на тренувальних даних = оптимістична оцінка,
# в реальності використовуй train_test_split (див. sklearn_overview_demo.py).
y_pred = model.predict(X)

print(f"model.fit(X, y)      — навчили на {len(X)} прикладах з мітками")
print(f"model.predict(X)     → {list(y_pred)}")
print(f"accuracy             → {accuracy_score(y, y_pred):.0%}")
print()

# Новий email — модель передбачає
new_email = pd.DataFrame({"довжина": [9], "кількість_посилань": [7], "великі_літери_%": [50]})
pred = model.predict(new_email)[0]
print(f"Новий email: довжина=9, посилань=7, великі літери=50%")
print(f"Передбачення: {'SPAM' if pred == 1 else 'НЕ SPAM'}")
print()

# ══════════════════════════════════════════════════════════════
# 2. UNSUPERVISED LEARNING
# ══════════════════════════════════════════════════════════════
# Правильних відповідей НЕМАЄ. Модель сама знаходить структуру.
# Як дитина яка сортує іграшки — сама вирішує що схоже на що.
#
# Приклади: кластеризація клієнтів, embeddings, anomaly detection

print("=" * 65)
print("2. UNSUPERVISED LEARNING — міток немає")
print("=" * 65)
print()

# Дані: клієнти магазину (без міток!)
clients = pd.DataFrame({
    "вік":              [22, 25, 23, 55, 58, 52, 35, 38, 33],
    "витрати_на_місяць": [200, 300, 250, 800, 900, 750, 1500, 1800, 1600],
})

print("Задача: сегментація клієнтів")
print("Дані: вік і витрати, БЕЗ міток — не знаємо скільки груп")
print(clients.to_string(index=False))
print()

# KMeans — знаходить кластери сам
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(clients)  # ← немає y!

print(f"kmeans.fit_predict(X)  — знайшов 3 кластери БЕЗ міток")
print()

clients["кластер"] = clusters
for c in sorted(clients["кластер"].unique()):
    group = clients[clients["кластер"] == c]
    avg_age = group["вік"].mean()
    avg_spend = group["витрати_на_місяць"].mean()
    print(f"  Кластер {c}: вік ~{avg_age:.0f}, витрати ~{avg_spend:.0f} грн"
          f"  ({len(group)} клієнтів)")

print()
print("Модель сама знайшла групи — ми тільки сказали 'шукай 3 кластери'")
print()

# ══════════════════════════════════════════════════════════════
# 3. REINFORCEMENT LEARNING
# ══════════════════════════════════════════════════════════════
# Немає датасету. Агент робить дії і отримує нагороду або штраф.
# Вчиться максимізувати нагороду через досвід.
# Як собака яка вчить команди — treat за правильну дію.
#
# Приклади: RLHF в LLM, ігри (AlphaGo), роботи, рекомендації

print("=" * 65)
print("3. REINFORCEMENT LEARNING — нагорода/штраф")
print("=" * 65)
print()

# Простий приклад: агент шукає вихід з лабіринту
# Середовище: лінійний коридор [0, 1, 2, 3, 4]
# Агент починає з позиції 0, вихід на позиції 4
# Дії: "вправо" (+1) або "вліво" (-1)
# Нагорода: +10 за вихід, -1 за кожен крок

print("Задача: агент шукає вихід з коридору")
print("Середовище: [0, 1, 2, 3, 4]  — вихід на позиції 4")
print("Дії: вправо (+1) або вліво (-1)")
print("Нагорода: +10 за вихід, -1 за кожен крок")
print()

# Q-table: для кожної позиції зберігаємо очікувану нагороду за кожну дію
q_table = np.zeros((5, 2))  # 5 позицій × 2 дії (0=вліво, 1=вправо)

learning_rate = 0.5
discount = 0.9
episodes = 100

for episode in range(episodes):
    pos = 0  # починаємо з позиції 0

    for step in range(20):  # максимум 20 кроків
        # Вибираємо дію: рандомно (explore) або найкращу (exploit)
        if np.random.random() < 0.3:  # 30% explore
            action = np.random.randint(2)
        else:  # 70% exploit
            action = np.argmax(q_table[pos])

        # Виконуємо дію
        new_pos = pos + (1 if action == 1 else -1)
        new_pos = max(0, min(4, new_pos))  # не виходимо за межі

        # Отримуємо нагороду
        reward = 10 if new_pos == 4 else -1

        # Оновлюємо Q-table (вчимося з досвіду)
        best_next = np.max(q_table[new_pos])
        q_table[pos][action] += learning_rate * (
            reward + discount * best_next - q_table[pos][action]
        )

        pos = new_pos
        if pos == 4:
            break

# Результат: агент навчився стратегії
print(f"Після {episodes} епізодів агент навчився:")
print()
print(f"  {'позиція':<10} {'← вліво':<12} {'вправо →':<12} {'найкраща дія'}")
print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*15}")
for pos in range(5):
    left_val = q_table[pos][0]
    right_val = q_table[pos][1]
    best = "→ вправо" if right_val > left_val else "← вліво"
    if pos == 4:
        best = "ВИХІД!"
    print(f"  [{pos}]{'':<7} {left_val:<12.1f} {right_val:<12.1f} {best}")

print()

# Демонструємо навчену стратегію
print("Навчена стратегія:")
pos = 0
path = [pos]
for _ in range(10):
    action = np.argmax(q_table[pos])
    pos = pos + (1 if action == 1 else -1)
    pos = max(0, min(4, pos))
    path.append(pos)
    if pos == 4:
        break

print(f"  {' → '.join(str(p) for p in path)}  — знайшов вихід за {len(path)-1} кроки!")
print()

# ══════════════════════════════════════════════════════════════
# ПОРІВНЯННЯ
# ══════════════════════════════════════════════════════════════

print("=" * 65)
print("ПОРІВНЯННЯ")
print("=" * 65)
print("""
  Тип                    Дані                  Приклади
  ─────────────────────  ────────────────────  ─────────────────────
  Supervised             X + y (є мітки)       spam, fraud, scoring
  Unsupervised           X (без міток)         кластери, embeddings
  Reinforcement          немає датасету        RLHF, AlphaGo, роботи

  AI-інженеру найчастіше: Supervised (80%) + Unsupervised (embeddings)
  Reinforcement — це RLHF для fine-tuning LLM
""")
