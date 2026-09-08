# Feynt

Лаунчер для одной локальной модели: строка меню, чат и мастер первого запуска. Не менеджер
моделей — выбор ровно из двух вариантов, без поиска и без произвольных репозиториев.

Движок работает **внутри приложения** через MLX Swift (`mlx-swift-lm`): нет Python, нет
дочернего процесса, нет HTTP-сервера и порта.

## Что умеет

- Мастер первого запуска: проверка RAM и места на диске → выбор модели → загрузка недостающего.
- Каталог из двух моделей (`Qwen3.8-27B` стоковая и `Uncensored`) плюс общий драфтер.
  Уже лежащие на диске веса находятся до того,
  как приложение предложит что-то скачивать.
- Пункт в строке меню: состояние, live tok/s, принятые токены за шаг, загрузка/выгрузка
  модели, открытие чата и лога, таймаут простоя, выход.
- Чат: история в памяти, стриминг, отдельная сворачиваемая область «размышлений»,
  моноширинные блоки кода, кнопка «Стоп».
- Выгрузка по простою: ссылки на веса сбрасываются, кэш MLX очищается, ~16 ГБ возвращаются
  системе. Таймаут настраивается (60 с … 1 ч или «никогда»).
- Встроенный OpenAI-совместимый сервер на `127.0.0.1:19234` (порт настраивается).

## Endpoint

Поднимается вместе с моделью, слушает только loopback, без сторонних HTTP-зависимостей
(Network.framework). Запросы обрабатываются строго по одному — GPU не делится.

| Метод | Путь | Что делает |
|---|---|---|
| POST | `/v1/chat/completions` | обычный ответ и SSE при `stream: true` |
| GET | `/v1/models` | список моделей |
| GET | `/health` | `status`: `ok` / `loading` / `no_model` / `error` |
| GET | `/metrics` | `requests`, `completion_tokens`, `mean_decode_tokens_per_sec`, `mean_accept_len` |

Учитываются `messages`, `max_tokens`, `stream`, `temperature`,
`chat_template_kwargs.enable_thinking`; незнакомые поля игнорируются. Блок размышлений
приходит в `choices[0].message.reasoning_content` (в SSE — `delta.reasoning_content`).

```sh
curl http://127.0.0.1:19234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"привет"}],"max_tokens":256}'
```

## Сборка

```sh
swift build            # отладочная сборка
swift build -c release # релизная
```

Требуется macOS 14+, Xcode 16+ и соседний checkout `mlx-swift-lm`:

```
../mlx-swift-lm      # путь-зависимость, см. Package.swift
```

`mlx-swift` и Hugging Face-пакеты тянутся из сети автоматически.

## Сборка .app

```sh
./package-app.sh            # debug → build/Feynt.app
./package-app.sh release    # оптимизированная сборка
```

Подпись по умолчанию ad-hoc (`codesign -s -`) — приложение работает локально без Developer ID.
Для распространяемой сборки:

```sh
FEYNT_SIGN_IDENTITY="Developer ID Application: …" ./package-app.sh release
```

Скрипт идемпотентен: старый бандл удаляется целиком, ресурсные бандлы SwiftPM получают
минимальный `Info.plist` (иначе `codesign` их отвергает) и подписываются изнутри наружу.

## Тесты

Тестовой цели пока нет. Когда появится — запускать её через `xcodebuild`, а не `swift test`:
из SwiftPM-теста `mlx-swift` не находит свой metallib.

```sh
xcodebuild test -scheme Feynt -destination 'platform=macOS' -skipPackagePluginValidation
```

## Архитектура

`InferenceEngine` (`Sources/Feynt/Core/InferenceEngine.swift`) — шов между UI и тем, что
крутит веса. v1 реализован в `MLXEngine`: MTP-спекулятивное декодирование из `mlx-swift-lm`,
с откатом на обычную генерацию, если драфтер не поддержан. Более быстрый драфтер
подключается заменой реализации протокола — UI об этом не знает.

## Установка

```sh
brew tap random1st/feynt
brew trust random1st/feynt
brew install --cask feynt
```
