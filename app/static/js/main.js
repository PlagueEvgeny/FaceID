let systemStatus = {
  camera_ready: false,
  model_trained: false,
  is_collecting: false,
  collected_count: 0,
}

// Утилиты
const Utils = {
  // Показать уведомление
  showNotification: function (message, type = "info", duration = 5000) {
    const notification = document.createElement("div")
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`
    notification.style.cssText = `
            top: 20px; 
            right: 20px; 
            z-index: 9999; 
            min-width: 300px;
            max-width: 400px;
        `

    notification.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-${this.getIconForType(type)} me-2"></i>
                <div>${message}</div>
                <button type="button" class="btn-close ms-auto" data-bs-dismiss="alert"></button>
            </div>
        `

    document.body.appendChild(notification)

    // Автоматическое удаление
    setTimeout(() => {
      if (notification.parentNode) {
        notification.remove()
      }
    }, duration)

    return notification
  },

  // Получить иконку для типа уведомления
  getIconForType: (type) => {
    const icons = {
      success: "check-circle",
      danger: "exclamation-triangle",
      warning: "exclamation-triangle",
      info: "info-circle",
      primary: "info-circle",
    }
    return icons[type] || "info-circle"
  },

  // Форматировать время
  formatTime: (date) =>
    new Intl.DateTimeFormat("ru-RU", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(date),

  // Форматировать дату
  formatDate: (date) =>
    new Intl.DateTimeFormat("ru-RU", {
      year: "numeric",
      month: "long",
      day: "numeric",
    }).format(date),

  // Проверить поддержку камеры
  checkCameraSupport: () => navigator.mediaDevices && navigator.mediaDevices.getUserMedia,

  // Загрузить изображение как base64
  loadImageAsBase64: (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result.split(",")[1])
      reader.onerror = reject
      reader.readAsDataURL(file)
    }),

  // Валидация файла изображения
  validateImageFile: (file) => {
    const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/gif"]
    const maxSize = 10 * 1024 * 1024 // 10MB

    if (!validTypes.includes(file.type)) {
      throw new Error("Неподдерживаемый тип файла. Используйте JPEG, PNG или GIF.")
    }

    if (file.size > maxSize) {
      throw new Error("Файл слишком большой. Максимальный размер: 10MB.")
    }

    return true
  },
}

// API клиент
const API = {
  baseUrl: "",

  // Выполнить GET запрос
  get: async function (endpoint) {
    try {
      const response = await fetch(this.baseUrl + endpoint)
      return await response.json()
    } catch (error) {
      console.error("API GET Error:", error)
      throw error
    }
  },

  // Выполнить POST запрос
  post: async function (endpoint, data = null, isFormData = false) {
    try {
      const options = {
        method: "POST",
        headers: {},
      }

      if (isFormData) {
        options.body = data
      } else {
        options.headers["Content-Type"] = "application/json"
        options.body = JSON.stringify(data)
      }

      const response = await fetch(this.baseUrl + endpoint, options)
      return await response.json()
    } catch (error) {
      console.error("API POST Error:", error)
      throw error
    }
  },

  // Получить статус системы
  getStatus: function () {
    return this.get("/get_status")
  },

  // Начать сбор данных
  startCollection: function (personName) {
    return this.post("/start_collection", { name: personName })
  },

  // Остановить сбор данных
  stopCollection: function () {
    return this.post("/stop_collection")
  },

  // Обучить модель
  trainModel: function () {
    return this.post("/train_model")
  },

  // Получить информацию о модели
  getModelInfo: function () {
    return this.get("/api/model_info")
  },

  // Распознать изображение
  recognizeImage: function (imageFile) {
    const formData = new FormData()
    formData.append("image", imageFile)
    return this.post("/api/recognize_image", formData, true)
  },

  // Обновить порог уверенности
  updateThreshold: function (threshold) {
    return this.post("/api/update_threshold", { threshold })
  },

  // Тест точности
  testAccuracy: function (testImagesPerPerson = 5) {
    return this.post("/api/test_accuracy", { test_images_per_person: testImagesPerPerson })
  },
}

// Менеджер статуса системы
const StatusManager = {
  updateInterval: null,
  callbacks: [],

  // Начать отслеживание статуса
  start: function (interval = 5000) {
    this.updateInterval = setInterval(() => {
      this.updateStatus()
    }, interval)

    // Первоначальное обновление
    this.updateStatus()
  },

  // Остановить отслеживание
  stop: function () {
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
      this.updateInterval = null
    }
  },

  // Обновить статус
  updateStatus: async function () {
    try {
      const status = await API.getStatus()
      systemStatus = status

      // Вызвать все колбэки
      this.callbacks.forEach((callback) => {
        try {
          callback(status)
        } catch (error) {
          console.error("Status callback error:", error)
        }
      })
    } catch (error) {
      console.error("Status update error:", error)
    }
  },

  // Добавить колбэк для обновления статуса
  addCallback: function (callback) {
    this.callbacks.push(callback)
  },

  // Удалить колбэк
  removeCallback: function (callback) {
    const index = this.callbacks.indexOf(callback)
    if (index > -1) {
      this.callbacks.splice(index, 1)
    }
  },
}

// Менеджер камеры
const CameraManager = {
  stream: null,
  video: null,

  // Инициализировать камеру
  init: async function (videoElement) {
    this.video = videoElement

    if (!Utils.checkCameraSupport()) {
      throw new Error("Камера не поддерживается в этом браузере")
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      })

      if (this.video) {
        this.video.srcObject = this.stream
      }

      return true
    } catch (error) {
      console.error("Camera init error:", error)
      throw new Error("Не удалось получить доступ к камере")
    }
  },

  // Остановить камеру
  stop: function () {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop())
      this.stream = null
    }

    if (this.video) {
      this.video.srcObject = null
    }
  },

  // Сделать снимок
  capture: function (canvas) {
    if (!this.video || !canvas) return null

    const context = canvas.getContext("2d")
    canvas.width = this.video.videoWidth
    canvas.height = this.video.videoHeight

    context.drawImage(this.video, 0, 0)
    return canvas.toDataURL("image/jpeg")
  },
}

// Обработчики событий для всего приложения
document.addEventListener("DOMContentLoaded", () => {
  // Инициализация Bootstrap tooltips
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
  const bootstrap = window.bootstrap // Declare the bootstrap variable
  tooltipTriggerList.map((tooltipTriggerEl) => new bootstrap.Tooltip(tooltipTriggerEl))

  // Инициализация Bootstrap popovers
  const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
  popoverTriggerList.map((popoverTriggerEl) => new bootstrap.Popover(popoverTriggerEl))

  // Автоматическое скрытие алертов
  const alerts = document.querySelectorAll(".alert:not(.alert-permanent)")
  alerts.forEach((alert) => {
    setTimeout(() => {
      if (alert.parentNode) {
        alert.classList.add("fade")
        setTimeout(() => alert.remove(), 150)
      }
    }, 5000)
  })

  // Плавная прокрутка для якорных ссылок
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (event) {
      event.preventDefault()
      const target = document.querySelector(this.getAttribute("href"))
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        })
      }
    })
  })

  // Обработка ошибок изображений
  document.querySelectorAll("img").forEach((img) => {
    img.addEventListener("error", function () {
      this.src = "/static/images/placeholder.png"
      this.alt = "Изображение не найдено"
    })
  })

  // Запуск менеджера статуса на главной странице
  if (document.getElementById("system-status")) {
    StatusManager.start()
  }
})

// Обработка ошибок JavaScript
window.addEventListener("error", (e) => {
  console.error("JavaScript Error:", e.error)

  // Показать пользователю уведомление об ошибке в режиме разработки
  if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    Utils.showNotification("Произошла ошибка JavaScript. Проверьте консоль.", "danger")
  }
})

// Обработка необработанных промисов
window.addEventListener("unhandledrejection", (e) => {
  console.error("Unhandled Promise Rejection:", e.reason)

  if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    Utils.showNotification("Произошла ошибка в промисе. Проверьте консоль.", "danger")
  }
})

// Экспорт для использования в других скриптах
window.FaceIDApp = {
  Utils,
  API,
  StatusManager,
  CameraManager,
  systemStatus,
}