/**
 * StockTradebyZ Web - 主JavaScript文件
 */

// 全局配置
const APP_CONFIG = {
    API_BASE_URL: '',
    DEFAULT_TIMEOUT: 30000,
    SSE_RETRY_DELAY: 3000
};

// 工具函数
const Utils = {
    /**
     * 格式化日期
     */
    formatDate(dateStr) {
        if (!dateStr) return '-';
        const date = new Date(dateStr);
        return date.toLocaleDateString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit'
        });
    },

    /**
     * 格式化日期时间
     */
    formatDateTime(dateStr) {
        if (!dateStr) return '-';
        const date = new Date(dateStr);
        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    },

    /**
     * 格式化数字
     */
    formatNumber(num, decimals = 2) {
        if (num === null || num === undefined) return '-';
        return Number(num).toFixed(decimals);
    },

    /**
     * 格式化大数字（万、亿）
     */
    formatLargeNumber(num) {
        if (num === null || num === undefined) return '-';

        if (Math.abs(num) >= 1e8) {
            return (num / 1e8).toFixed(2) + '亿';
        } else if (Math.abs(num) >= 1e4) {
            return (num / 1e4).toFixed(2) + '万';
        } else {
            return num.toFixed(2);
        }
    },

    /**
     * 格式化百分比
     */
    formatPercentage(num, decimals = 2) {
        if (num === null || num === undefined) return '-';
        return Number(num).toFixed(decimals) + '%';
    },

    /**
     * 显示Toast提示
     */
    showToast(message, type = 'info') {
        const bgClass = {
            'success': 'bg-success',
            'error': 'bg-danger',
            'warning': 'bg-warning',
            'info': 'bg-info'
        }[type] || 'bg-info';

        const toast = $(`
            <div class="toast align-items-center text-white ${bgClass} border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">${message}</div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `);

        // 添加到页面
        if ($('#toast-container').length === 0) {
            $('body').append('<div id="toast-container" class="toast-container position-fixed top-0 end-0 p-3"></div>');
        }

        $('#toast-container').append(toast);

        // 显示Toast
        const bsToast = new bootstrap.Toast(toast[0], { delay: 3000 });
        bsToast.show();

        // 自动移除
        toast.on('hidden.bs.toast', function() {
            $(this).remove();
        });
    },

    /**
     * 显示加载状态
     */
    showLoading(message = '加载中...') {
        if ($('#loading-overlay').length === 0) {
            $('body').append(`
                <div id="loading-overlay" class="position-fixed top-0 start-0 w-100 h-100 d-flex align-items-center justify-content-center" style="background: rgba(0,0,0,0.5); z-index: 9999;">
                    <div class="text-center text-white">
                        <div class="spinner-border mb-3" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p id="loading-message">${message}</p>
                    </div>
                </div>
            `);
        } else {
            $('#loading-message').text(message);
            $('#loading-overlay').show();
        }
    },

    /**
     * 隐藏加载状态
     */
    hideLoading() {
        $('#loading-overlay').fadeOut(300, function() {
            $(this).remove();
        });
    },

    /**
     * AJAX错误处理
     */
    handleAjaxError(xhr, status, error) {
        console.error('AJAX Error:', status, error);

        let message = '请求失败';
        if (xhr.responseJSON && xhr.responseJSON.error) {
            message = xhr.responseJSON.error;
        } else if (error) {
            message = error;
        }

        Utils.showToast(message, 'error');
    }
};

// API调用封装
const API = {
    /**
     * 通用GET请求
     */
    get(url, params = {}) {
        return $.ajax({
            url: APP_CONFIG.API_BASE_URL + url,
            method: 'GET',
            data: params,
            timeout: APP_CONFIG.DEFAULT_TIMEOUT
        });
    },

    /**
     * 通用POST请求
     */
    post(url, data = {}) {
        return $.ajax({
            url: APP_CONFIG.API_BASE_URL + url,
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(data),
            timeout: APP_CONFIG.DEFAULT_TIMEOUT
        });
    },

    /**
     * 通用DELETE请求
     */
    delete(url) {
        return $.ajax({
            url: APP_CONFIG.API_BASE_URL + url,
            method: 'DELETE',
            timeout: APP_CONFIG.DEFAULT_TIMEOUT
        });
    }
};

// SSE连接管理
class SSEConnection {
    constructor(url) {
        this.url = url;
        this.eventSource = null;
        this.onMessage = null;
        this.onError = null;
        this.onClose = null;
    }

    connect() {
        this.eventSource = new EventSource(this.url);

        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (this.onMessage) {
                    this.onMessage(data);
                }
            } catch (e) {
                console.error('SSE message parse error:', e);
            }
        };

        this.eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            this.close();

            if (this.onError) {
                this.onError(error);
            }
        };
    }

    close() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;

            if (this.onClose) {
                this.onClose();
            }
        }
    }
}

// 页面初始化
$(document).ready(function() {
    // 初始化所有tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // 初始化所有popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // 全局AJAX错误处理
    $(document).ajaxError(function(event, xhr, settings, error) {
        if (settings.suppressError) return;
        Utils.handleAjaxError(xhr, 'error', error);
    });
});

// 暴露到全局
window.APP_CONFIG = APP_CONFIG;
window.Utils = Utils;
window.API = API;
window.SSEConnection = SSEConnection;
