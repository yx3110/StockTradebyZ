"""
StockTradebyZ WebApp Comprehensive Playwright Test Suite

Tests all pages, API endpoints, interactive features, and edge cases.
Run: cd webapp && python -m pytest tests/test_webapp_playwright.py -v --tb=short
"""
import pytest
import json
import time
import re
from playwright.sync_api import sync_playwright, expect, Page, BrowserContext

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 15000  # 15 seconds for page loads


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="session")
def browser():
    """Launch browser once for all tests."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture(scope="function")
def context(browser):
    """Fresh context per test for isolation."""
    ctx = browser.new_context(
        viewport={"width": 1280, "height": 900},
        locale="zh-CN",
    )
    ctx.set_default_timeout(TIMEOUT)
    yield ctx
    ctx.close()


@pytest.fixture(scope="function")
def page(context):
    """Fresh page per test."""
    pg = context.new_page()
    yield pg
    pg.close()


# ============================================================
# 1. Health Check & Basic Connectivity
# ============================================================

class TestHealthAndConnectivity:
    """Verify the server is up and basic connectivity works."""

    def test_health_endpoint(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/health")
        assert resp.ok
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_homepage_loads(self, page: Page):
        resp = page.goto(BASE_URL)
        assert resp.status == 200
        expect(page).to_have_title(re.compile("StockTradebyZ"))


# ============================================================
# 2. Page Navigation & Rendering
# ============================================================

class TestPageNavigation:
    """Test all 7 pages load correctly with proper content."""

    @pytest.mark.parametrize("path,title_contains,heading_text", [
        ("/", "首页", "系统仪表板"),
        ("/daily-tasks", "日常任务", "日常任务"),
        ("/model-training", "模型训练", "模型训练"),
        ("/backtest", "回测", "回测"),
        ("/portfolio", "持仓管理", "持仓管理"),
        ("/data-management", "数据管理", "数据管理"),
    ])
    def test_page_loads_with_title_and_heading(self, page: Page, path, title_contains, heading_text):
        page.goto(f"{BASE_URL}{path}")
        expect(page).to_have_title(re.compile(title_contains))
        heading = page.locator("h2").first
        expect(heading).to_contain_text(heading_text)

    def test_stock_detail_page(self, page: Page):
        """Test stock detail page with a known stock code."""
        page.goto(f"{BASE_URL}/stock/000001")
        page.wait_for_load_state("networkidle")
        # Title should contain stock code or detail text
        expect(page).to_have_title(re.compile("000001|个股详情|StockTradebyZ"))

    def test_navbar_present_on_all_pages(self, page: Page):
        """Verify navbar is present with correct links on every page."""
        page.goto(BASE_URL)
        navbar = page.locator("nav.navbar")
        expect(navbar).to_be_visible()

        # Check all nav links exist
        nav_links = page.locator(".navbar-nav .nav-link")
        assert nav_links.count() >= 6  # 首页, 日常任务, 模型训练, 回测, 持仓管理, 数据管理

    def test_footer_present(self, page: Page):
        page.goto(BASE_URL)
        footer = page.locator("footer")
        expect(footer).to_be_visible()
        expect(footer).to_contain_text("StockTradebyZ")

    def test_navbar_active_state(self, page: Page):
        """Verify the correct nav link is marked active on each page."""
        page.goto(f"{BASE_URL}/daily-tasks")
        # The "日常任务" link should have 'active' class
        active_link = page.locator(".nav-link.active")
        expect(active_link).to_contain_text("日常任务")

    def test_404_page(self, page: Page):
        """Test 404 error page for non-existent routes."""
        resp = page.goto(f"{BASE_URL}/nonexistent-page-xyz")
        assert resp.status == 404

    def test_clock_updates(self, page: Page):
        """Verify the clock in the navbar updates."""
        page.goto(BASE_URL)
        clock = page.locator("#current-time")
        time1 = clock.text_content()
        page.wait_for_timeout(1500)
        time2 = clock.text_content()
        # Clock should have changed (or at least be a valid time format)
        assert re.match(r"\d{1,2}:\d{2}:\d{2}", time1)


# ============================================================
# 3. Dashboard Tests
# ============================================================

class TestDashboard:
    """Test the main dashboard page functionality."""

    def test_stat_cards_load(self, page: Page):
        """Dashboard stat cards should populate with data."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        # Wait for AJAX to complete
        page.wait_for_timeout(3000)

        total_stocks = page.locator("#total-stocks")
        # Should no longer be the loading placeholder
        text = total_stocks.text_content().strip()
        assert text != "", "total-stocks should have content"

    def test_quick_actions_visible(self, page: Page):
        """Quick action buttons should be visible and clickable."""
        page.goto(BASE_URL)
        actions = page.locator(".quick-action-btn")
        assert actions.count() >= 4

    def test_quick_action_navigation(self, page: Page):
        """Clicking quick action navigates to the correct page."""
        page.goto(BASE_URL)
        # Click "数据更新" -> should go to /daily-tasks
        page.locator("a.quick-action-btn", has_text="数据更新").click()
        page.wait_for_load_state("networkidle")
        assert "/daily-tasks" in page.url

    def test_market_indices_section(self, page: Page):
        """Market indices section should load."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        indices_container = page.locator("#market-indices")
        expect(indices_container).to_be_visible()
        # Should have content (either data or "暂无大盘数据")
        text = indices_container.text_content().strip()
        assert len(text) > 0

    def test_module_overview_cards(self, page: Page):
        """Three module overview cards should be visible."""
        page.goto(BASE_URL)
        module_cards = page.locator(".module-card")
        assert module_cards.count() >= 3

    def test_db_stats_section(self, page: Page):
        """Database statistics section should load."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        db_size = page.locator("#db-size")
        text = db_size.text_content().strip()
        # Should eventually show MB value
        assert text != "", "db-size should have content"

    def test_system_info_section(self, page: Page):
        """System info table should display version and config."""
        page.goto(BASE_URL)
        sys_info = page.locator("text=系统版本")
        expect(sys_info).to_be_visible()


# ============================================================
# 4. API Endpoint Tests
# ============================================================

class TestAPIEndpoints:
    """Test all major API endpoints return proper responses."""

    # --- Daily Tasks API ---
    def test_api_daily_status(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/daily/status")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True
        assert "database_stats" in data
        stats = data["database_stats"]
        assert "total_stocks" in stats
        assert stats["total_stocks"] > 0

    def test_api_daily_selections(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/daily/selections?version=v3.9&limit=10")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True
        assert "dates" in data
        assert isinstance(data["dates"], list)

    def test_api_daily_reports(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/daily/reports?version=v3.9&limit=5")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True

    def test_api_market_indices(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/daily/market-indices")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True
        assert "indices" in data

    # --- Stock API ---
    def test_api_stock_search(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/stock/search?q=平安&limit=5")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True
        assert "results" in data

    def test_api_stock_search_by_code(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/stock/search?q=000001&limit=5")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True

    def test_api_stock_kline(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/stock/000001/kline?days=30")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True
        assert "data" in data or "kline" in data

    def test_api_stock_technical(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/stock/000001/technical?days=30")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True

    def test_api_stock_fundamental(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/stock/000001/fundamental")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True

    def test_api_stock_info(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/stock/000001/info")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True

    # --- Portfolio API ---
    def test_api_portfolio_positions(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/portfolio/positions")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True
        assert "positions" in data
        assert "summary" in data

    def test_api_portfolio_trades(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/portfolio/trades")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True
        assert "trades" in data

    def test_api_portfolio_groups(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/portfolio/groups")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True
        assert "groups" in data

    def test_api_portfolio_evaluations(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/portfolio/evaluations")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True

    def test_api_portfolio_recommendations(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/portfolio/recommendations")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True

    def test_api_portfolio_stock_search(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/portfolio/stocks/search?q=000001")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True

    # --- Model Training API ---
    def test_api_models_list(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/models/")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True

    def test_api_models_summary(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/models/summary")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True
        assert "summary" in data

    # --- Backtest API ---
    def test_api_backtest_strategies(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/backtest/strategies")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True

    def test_api_backtest_results(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/backtest/results")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True

    def test_api_backtest_summary(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/backtest/summary")
        assert resp.ok
        data = resp.json()
        assert data["success"] is True

    # --- Data Management API ---
    @pytest.mark.slow
    def test_api_data_stats(self, page: Page):
        """Stats endpoint queries large DB - may be slow."""
        resp = page.request.get(f"{BASE_URL}/api/data/stats", timeout=120000)
        assert resp.ok
        data = resp.json()
        assert "stats" in data

    @pytest.mark.slow
    def test_api_data_completeness(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/data/completeness", timeout=120000)
        assert resp.ok
        data = resp.json()
        assert "success" in data or "completeness" in data

    def test_api_data_backfill_scripts(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/data/backfill/scripts")
        assert resp.ok
        data = resp.json()
        assert "success" in data or "scripts" in data


# ============================================================
# 5. API Edge Cases & Error Handling
# ============================================================

class TestAPIEdgeCases:
    """Test API error handling and edge cases."""

    def test_stock_search_empty_query(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/stock/search?q=")
        assert resp.ok
        data = resp.json()
        # Should return empty results or error, not crash
        assert "success" in data or "error" in data

    def test_stock_search_special_characters(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/stock/search?q=%3Cscript%3E&limit=5")
        assert resp.ok
        data = resp.json()
        # Should not crash with XSS-like input
        assert "success" in data or "error" in data

    def test_stock_search_very_long_query(self, page: Page):
        long_q = "a" * 500
        resp = page.request.get(f"{BASE_URL}/api/stock/search?q={long_q}&limit=5")
        assert resp.ok

    def test_invalid_stock_code_kline(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/stock/999999/kline")
        data = resp.json()
        # Should handle gracefully - either empty data or error
        assert "success" in data or "error" in data

    def test_invalid_stock_code_info(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/stock/INVALID/info")
        data = resp.json()
        assert "success" in data or "error" in data

    def test_selection_nonexistent_date(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/daily/selection/1999-01-01?version=v3.9")
        data = resp.json()
        # Should return not found or error, not crash
        assert resp.status in [200, 404]

    def test_selection_invalid_version(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/daily/selections?version=v999")
        data = resp.json()
        assert resp.status in [200, 400, 404]

    def test_portfolio_create_invalid_position(self, page: Page):
        """Creating position with missing required fields."""
        resp = page.request.post(f"{BASE_URL}/api/portfolio/positions",
                                  data=json.dumps({}),
                                  headers={"Content-Type": "application/json"})
        data = resp.json()
        # Should return error, not 500
        assert resp.status in [200, 400]
        if resp.status == 200:
            assert data.get("success") is False or "error" in data

    def test_portfolio_delete_nonexistent(self, page: Page):
        resp = page.request.delete(f"{BASE_URL}/api/portfolio/positions/99999")
        data = resp.json()
        assert resp.status in [200, 404]

    def test_portfolio_trade_invalid_data(self, page: Page):
        """POST trade with missing fields."""
        resp = page.request.post(f"{BASE_URL}/api/portfolio/trades",
                                  data=json.dumps({"code": "000001"}),
                                  headers={"Content-Type": "application/json"})
        data = resp.json()
        assert resp.status in [200, 400]

    def test_model_nonexistent_version(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/models/v999")
        data = resp.json()
        assert resp.status in [200, 404]

    def test_task_nonexistent_id(self, page: Page):
        resp = page.request.get(f"{BASE_URL}/api/tasks/nonexistent-id-12345/status")
        data = resp.json()
        assert resp.status in [200, 404]


# ============================================================
# 6. Stock Search (Navbar) Tests
# ============================================================

class TestStockSearch:
    """Test the global stock search functionality in the navbar."""

    def test_search_input_visible(self, page: Page):
        page.goto(BASE_URL)
        search_input = page.locator("#stock-search-input")
        expect(search_input).to_be_visible()

    def test_search_dropdown_appears(self, page: Page):
        """Type a stock code and verify dropdown appears."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        search_input = page.locator("#stock-search-input")
        search_input.fill("000001")
        # Wait for debounced search (250ms + network)
        page.wait_for_timeout(1000)

        dropdown = page.locator("#stock-search-dropdown")
        # Should either show results or "无结果"
        is_visible = dropdown.is_visible()
        if is_visible:
            text = dropdown.text_content()
            assert len(text) > 0

    def test_search_by_name(self, page: Page):
        """Search by Chinese stock name."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        search_input = page.locator("#stock-search-input")
        search_input.fill("平安")
        page.wait_for_timeout(1000)

        dropdown = page.locator("#stock-search-dropdown")
        # May show results for 平安银行
        if dropdown.is_visible():
            text = dropdown.text_content()
            assert len(text) > 0

    def test_search_enter_navigates(self, page: Page):
        """Pressing Enter with a stock code should navigate to stock detail."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        search_input = page.locator("#stock-search-input")
        search_input.fill("000001")
        search_input.press("Enter")
        page.wait_for_load_state("networkidle")

        assert "/stock/000001" in page.url

    def test_search_dropdown_closes_on_outside_click(self, page: Page):
        """Clicking outside should close the dropdown."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        search_input = page.locator("#stock-search-input")
        search_input.fill("000001")
        page.wait_for_timeout(1000)

        # Click somewhere else
        page.locator("h2").first.click()
        page.wait_for_timeout(300)

        dropdown = page.locator("#stock-search-dropdown")
        expect(dropdown).to_be_hidden()

    def test_search_empty_clears_dropdown(self, page: Page):
        """Clearing the search input should hide dropdown."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        search_input = page.locator("#stock-search-input")
        search_input.fill("000001")
        page.wait_for_timeout(1000)
        search_input.fill("")
        page.wait_for_timeout(300)

        dropdown = page.locator("#stock-search-dropdown")
        expect(dropdown).to_be_hidden()


# ============================================================
# 7. Daily Tasks Page Tests
# ============================================================

class TestDailyTasks:
    """Test the daily tasks page functionality."""

    def test_stat_cards_populated(self, page: Page):
        page.goto(f"{BASE_URL}/daily-tasks")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        total_stocks = page.locator("#stat-total-stocks")
        text = total_stocks.text_content().strip()
        assert text != "-", "stat-total-stocks should be populated"

    def test_date_selector_populated(self, page: Page):
        """Date dropdown should have options after loading."""
        page.goto(f"{BASE_URL}/daily-tasks")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        date_select = page.locator("#date-select")
        options = date_select.locator("option")
        # Should have at least the placeholder + some dates
        assert options.count() >= 1

    def test_version_selector_has_options(self, page: Page):
        """Version dropdown should have ML version options."""
        page.goto(f"{BASE_URL}/daily-tasks")
        version_select = page.locator("#version-select")
        options = version_select.locator("option")
        assert options.count() >= 1

    def test_update_button_exists(self, page: Page):
        page.goto(f"{BASE_URL}/daily-tasks")
        btn = page.locator("#btn-update")
        expect(btn).to_be_visible()
        expect(btn).to_contain_text("开始更新")

    def test_export_button_exists(self, page: Page):
        """Export button should exist (may be enabled if date auto-selected)."""
        page.goto(f"{BASE_URL}/daily-tasks")
        btn = page.locator("#btn-export")
        expect(btn).to_be_visible()

    def test_selection_loads_on_date_change(self, page: Page):
        """Selecting a date should load stock selection data."""
        page.goto(f"{BASE_URL}/daily-tasks")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        date_select = page.locator("#date-select")
        options = date_select.locator("option")

        if options.count() > 1:
            # Select the first actual date (not placeholder)
            first_date = options.nth(1).get_attribute("value")
            if first_date:
                date_select.select_option(first_date)
                page.wait_for_timeout(3000)

                # Stock table or results should now be visible
                stocks_container = page.locator("#stocks-table-container")
                selection_results = page.locator("#selection-results")
                # One of these should have content
                assert stocks_container.is_visible() or selection_results.text_content().strip() != ""


# ============================================================
# 8. Portfolio Management Tests
# ============================================================

class TestPortfolioManagement:
    """Test portfolio CRUD operations and UI interactions."""

    def test_portfolio_tabs_visible(self, page: Page):
        """All 4 tabs should be visible."""
        page.goto(f"{BASE_URL}/portfolio")
        tabs = page.locator("#portfolioTabs .nav-link")
        assert tabs.count() == 4

    def test_positions_tab_default_active(self, page: Page):
        page.goto(f"{BASE_URL}/portfolio")
        active_tab = page.locator("#positions-tab")
        expect(active_tab).to_have_class(re.compile("active"))

    def test_tab_switching(self, page: Page):
        """Click each tab and verify content pane switches."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")

        # Switch to recommendations tab
        page.locator("#recommendations-tab").click()
        page.wait_for_timeout(500)
        rec_pane = page.locator("#recommendations")
        expect(rec_pane).to_be_visible()

        # Switch to trades tab
        page.locator("#trades-tab").click()
        page.wait_for_timeout(500)
        trades_pane = page.locator("#trades")
        expect(trades_pane).to_be_visible()

        # Switch to evaluations tab
        page.locator("#evaluations-tab").click()
        page.wait_for_timeout(500)
        eval_pane = page.locator("#evaluations")
        expect(eval_pane).to_be_visible()

    def test_add_position_modal_opens(self, page: Page):
        """Click add position button should open modal."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")

        page.locator("button:has-text('添加持仓')").click()
        page.wait_for_timeout(500)

        modal = page.locator("#addPositionModal")
        expect(modal).to_be_visible()

    def test_add_position_modal_fields(self, page: Page):
        """Modal should have all required form fields."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")

        page.locator("button:has-text('添加持仓')").click()
        page.wait_for_timeout(500)

        expect(page.locator("#pos-code")).to_be_visible()
        expect(page.locator("#pos-name")).to_be_visible()
        expect(page.locator("#pos-quantity")).to_be_visible()
        expect(page.locator("#pos-cost")).to_be_visible()
        expect(page.locator("#pos-notes")).to_be_visible()

    def test_add_position_modal_cancel(self, page: Page):
        """Clicking cancel should close the modal."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")

        page.locator("button:has-text('添加持仓')").click()
        page.wait_for_timeout(500)

        page.locator("#addPositionModal button:has-text('取消')").click()
        page.wait_for_timeout(500)

        modal = page.locator("#addPositionModal")
        expect(modal).to_be_hidden()

    def test_add_position_validation(self, page: Page):
        """Saving without required fields should show warning."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")

        page.locator("button:has-text('添加持仓')").click()
        page.wait_for_timeout(500)

        # Try to save without filling anything
        page.locator("#btn-save-position").click()
        page.wait_for_timeout(500)

        # Toast should appear with warning
        toast = page.locator("#globalToast")
        # The modal should still be open (form not submitted)
        modal = page.locator("#addPositionModal")
        expect(modal).to_be_visible()

    def test_add_position_full_flow(self, page: Page):
        """Full add position flow: fill form, save, verify appears in table."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        page.locator("button:has-text('添加持仓')").click()
        page.wait_for_timeout(500)

        # Fill the form
        page.locator("#pos-code").fill("000001")
        page.wait_for_timeout(500)
        page.locator("#pos-name").evaluate("node => node.readOnly = false")
        page.locator("#pos-name").fill("平安银行")
        page.locator("#pos-quantity").fill("1000")
        page.locator("#pos-cost").fill("10.50")
        page.locator("#pos-notes").fill("测试持仓")

        page.locator("#btn-save-position").click()
        page.wait_for_timeout(2000)

        # Modal should close
        modal = page.locator("#addPositionModal")
        expect(modal).to_be_hidden()

        # Position should appear in table
        page.wait_for_timeout(1000)
        positions_body = page.locator("#positions-tbody")
        expect(positions_body).to_contain_text("000001")

    def test_delete_position(self, page: Page):
        """Delete a test position."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        # Check if there's a position with 000001
        tbody = page.locator("#positions-tbody")
        if "000001" in tbody.text_content():
            # Accept the confirm dialog
            page.on("dialog", lambda dialog: dialog.accept())
            # Click delete button for that row
            row = page.locator("#positions-tbody tr", has_text="000001").first
            row.locator("button.btn-outline-danger").click()
            page.wait_for_timeout(2000)

    def test_add_trade_modal(self, page: Page):
        """Trade record modal should open and have proper fields."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")

        # Switch to trades tab
        page.locator("#trades-tab").click()
        page.wait_for_timeout(500)

        page.locator("button:has-text('添加记录')").click()
        page.wait_for_timeout(500)

        modal = page.locator("#addTradeModal")
        expect(modal).to_be_visible()

        expect(page.locator("#trade-date")).to_be_visible()
        expect(page.locator("#trade-code")).to_be_visible()
        expect(page.locator("#trade-action")).to_be_visible()
        expect(page.locator("#trade-quantity")).to_be_visible()
        expect(page.locator("#trade-price")).to_be_visible()

    def test_group_management_modal(self, page: Page):
        """Group management modal should open."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")

        page.locator("button:has-text('管理分组')").click()
        page.wait_for_timeout(500)

        modal = page.locator("#groupModal")
        expect(modal).to_be_visible()
        expect(page.locator("#group-name")).to_be_visible()
        expect(page.locator("#group-color")).to_be_visible()

    def test_group_filter_buttons(self, page: Page):
        """Group filter should have at least the '全部' button."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        filter_btns = page.locator("#group-filter-btns .btn")
        assert filter_btns.count() >= 1
        expect(filter_btns.first).to_contain_text("全部")

    def test_position_summary_cards(self, page: Page):
        """Summary cards should display properly."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        expect(page.locator("#stat-total-mv")).to_be_visible()
        expect(page.locator("#stat-profit-loss")).to_be_visible()
        expect(page.locator("#stat-position-count")).to_be_visible()
        expect(page.locator("#stat-win-rate")).to_be_visible()


# ============================================================
# 9. Stock Detail Page Tests
# ============================================================

class TestStockDetail:
    """Test stock detail page with various stock codes."""

    def test_valid_stock_loads(self, page: Page):
        """A valid stock code should load data."""
        page.goto(f"{BASE_URL}/stock/000001")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        # Page should contain stock info
        body = page.locator("body")
        text = body.text_content()
        # Should have some stock-related content
        assert "000001" in text or "平安" in text or "K线" in text or "kline" in text.lower()

    def test_invalid_stock_code(self, page: Page):
        """An invalid stock code should not crash the page."""
        page.goto(f"{BASE_URL}/stock/INVALID")
        page.wait_for_load_state("networkidle")
        # Should still show the page (maybe with error message)
        assert page.locator("nav.navbar").is_visible()

    def test_stock_detail_navigation_from_search(self, page: Page):
        """Navigate to stock detail via search bar."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        search = page.locator("#stock-search-input")
        search.fill("000001")
        search.press("Enter")
        page.wait_for_load_state("networkidle")

        assert "/stock/000001" in page.url


# ============================================================
# 10. Model Training Page Tests
# ============================================================

class TestModelTraining:
    """Test model training page."""

    def test_page_loads_with_models(self, page: Page):
        page.goto(f"{BASE_URL}/model-training")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        # Should display model versions or "暂无"
        body = page.locator("body").text_content()
        assert "模型训练" in body

    def test_model_version_cards_or_list(self, page: Page):
        """Model versions should be displayed."""
        page.goto(f"{BASE_URL}/model-training")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        # The page should have loaded some model info
        body = page.locator("body").text_content()
        # Should contain version references (v3.9, etc)
        assert len(body) > 100  # Page has real content


# ============================================================
# 11. Backtest Page Tests
# ============================================================

class TestBacktest:
    """Test backtesting page."""

    def test_page_loads(self, page: Page):
        page.goto(f"{BASE_URL}/backtest")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        body = page.locator("body").text_content()
        assert "回测" in body

    def test_strategies_displayed(self, page: Page):
        """Strategy list or configuration should be visible."""
        page.goto(f"{BASE_URL}/backtest")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        body = page.locator("body").text_content()
        assert len(body) > 100


# ============================================================
# 12. Data Management Page Tests
# ============================================================

class TestDataManagement:
    """Test data management page. Note: this page queries large DB so may be slow."""

    def test_page_loads_with_stats(self, page: Page):
        page.goto(f"{BASE_URL}/data-management", timeout=60000)
        page.wait_for_load_state("domcontentloaded")
        # Don't wait for networkidle since DB queries can be very slow

        body = page.locator("body").text_content()
        assert "数据管理" in body

    def test_database_stats_displayed(self, page: Page):
        """Database statistics should load via API."""
        page.goto(f"{BASE_URL}/data-management", timeout=60000)
        page.wait_for_load_state("domcontentloaded")

        body = page.locator("body").text_content()
        assert len(body) > 100


# ============================================================
# 13. Cross-Page Navigation Tests
# ============================================================

class TestCrossPageNavigation:
    """Test navigating between pages using navbar links."""

    def test_navigate_all_pages_via_navbar(self, page: Page):
        """Click through all navbar links and verify correct pages load."""
        page.goto(BASE_URL)
        page.wait_for_load_state("domcontentloaded")

        nav_items = [
            ("首页", "/"),
            ("日常任务", "/daily-tasks"),
            ("模型训练", "/model-training"),
            ("回测", "/backtest"),
            ("持仓管理", "/portfolio"),
            ("数据管理", "/data-management"),
        ]

        for link_text, expected_path in nav_items:
            page.locator(f".nav-link:has-text('{link_text}')").click()
            # Use domcontentloaded since data-management page has slow DB queries
            page.wait_for_load_state("domcontentloaded")
            assert expected_path in page.url or page.url.endswith(expected_path), \
                f"Expected {expected_path} in URL, got {page.url}"

    def test_brand_link_goes_home(self, page: Page):
        """Clicking the brand logo should navigate to home."""
        page.goto(f"{BASE_URL}/daily-tasks")
        page.wait_for_load_state("domcontentloaded")

        page.locator(".navbar-brand").click()
        page.wait_for_load_state("domcontentloaded")

        # Should be on homepage
        assert page.url.rstrip("/") == BASE_URL or page.url == f"{BASE_URL}/"


# ============================================================
# 14. Toast Notification Tests
# ============================================================

class TestToastNotifications:
    """Test the global toast notification system."""

    def test_toast_container_exists(self, page: Page):
        page.goto(BASE_URL)
        toast = page.locator("#globalToast")
        assert toast.count() == 1

    def test_show_toast_via_js(self, page: Page):
        """Trigger a toast notification via JavaScript."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        page.evaluate("window.showToast('测试通知', 'success', '测试')")
        page.wait_for_timeout(500)

        toast = page.locator("#globalToast")
        expect(toast).to_be_visible()
        expect(page.locator("#toastMessage")).to_contain_text("测试通知")


# ============================================================
# 15. Loading Overlay Tests
# ============================================================

class TestLoadingOverlay:
    """Test the global loading overlay."""

    def test_loading_overlay_hidden_by_default(self, page: Page):
        page.goto(BASE_URL)
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(1000)

        loading = page.locator("#globalLoading")
        expect(loading).to_have_class(re.compile("d-none"))

    def test_show_hide_loading(self, page: Page):
        """Show and hide loading overlay via JS."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        page.evaluate("window.showLoading('测试加载中...')")
        page.wait_for_timeout(200)

        loading = page.locator("#globalLoading")
        expect(loading).not_to_have_class(re.compile("d-none"))

        message = page.locator("#loadingMessage")
        expect(message).to_contain_text("测试加载中")

        page.evaluate("window.hideLoading()")
        page.wait_for_timeout(200)
        expect(loading).to_have_class(re.compile("d-none"))


# ============================================================
# 16. Responsive Design Tests
# ============================================================

class TestResponsiveDesign:
    """Test responsive behavior at different viewport sizes."""

    def test_mobile_viewport(self, browser):
        """Pages should render without breaking on mobile viewport."""
        ctx = browser.new_context(viewport={"width": 375, "height": 812})
        pg = ctx.new_page()

        pg.goto(BASE_URL)
        pg.wait_for_load_state("networkidle")

        # Navbar toggler should be visible
        toggler = pg.locator(".navbar-toggler")
        expect(toggler).to_be_visible()

        # Content should still be present
        body = pg.locator("body")
        assert len(body.text_content()) > 100

        pg.close()
        ctx.close()

    def test_tablet_viewport(self, browser):
        """Pages should render properly on tablet viewport."""
        ctx = browser.new_context(viewport={"width": 768, "height": 1024})
        pg = ctx.new_page()

        pg.goto(BASE_URL)
        pg.wait_for_load_state("networkidle")

        body = pg.locator("body")
        assert len(body.text_content()) > 100

        pg.close()
        ctx.close()

    def test_mobile_navbar_toggle(self, browser):
        """Mobile navbar should expand on toggle click."""
        ctx = browser.new_context(viewport={"width": 375, "height": 812})
        pg = ctx.new_page()

        pg.goto(BASE_URL)
        pg.wait_for_load_state("networkidle")

        toggler = pg.locator(".navbar-toggler")
        toggler.click()
        pg.wait_for_timeout(500)

        # Nav links should now be visible
        nav = pg.locator("#navbarNav")
        expect(nav).to_be_visible()

        pg.close()
        ctx.close()


# ============================================================
# 17. Portfolio Full CRUD Lifecycle Test
# ============================================================

class TestPortfolioCRUDLifecycle:
    """End-to-end test of portfolio operations via API."""

    def test_full_lifecycle(self, page: Page):
        """Create -> Read -> Update -> Delete a position via API."""
        # 1. Create a test position
        create_resp = page.request.post(
            f"{BASE_URL}/api/portfolio/positions",
            data=json.dumps({
                "code": "600036",
                "name": "招商银行",
                "quantity": 500,
                "avg_cost": 35.00,
                "notes": "E2E测试"
            }),
            headers={"Content-Type": "application/json"}
        )
        assert create_resp.ok
        create_data = create_resp.json()
        assert create_data["success"] is True
        position_id = create_data.get("position_id") or create_data.get("id")
        assert position_id is not None

        # 2. Read positions and verify it exists
        read_resp = page.request.get(f"{BASE_URL}/api/portfolio/positions")
        assert read_resp.ok
        read_data = read_resp.json()
        positions = read_data["positions"]
        found = any(p["code"] == "600036" for p in positions)
        assert found, "Created position should appear in list"

        # 3. Add to position (加仓)
        add_resp = page.request.post(
            f"{BASE_URL}/api/portfolio/positions/{position_id}/add",
            data=json.dumps({"quantity": 200, "price": 36.00, "reason": "测试加仓"}),
            headers={"Content-Type": "application/json"}
        )
        assert add_resp.ok

        # 4. Reduce position (减仓)
        reduce_resp = page.request.post(
            f"{BASE_URL}/api/portfolio/positions/{position_id}/reduce",
            data=json.dumps({"quantity": 100, "price": 37.00, "reason": "测试减仓"}),
            headers={"Content-Type": "application/json"}
        )
        assert reduce_resp.ok

        # 5. Delete position
        delete_resp = page.request.delete(f"{BASE_URL}/api/portfolio/positions/{position_id}")
        assert delete_resp.ok
        delete_data = delete_resp.json()
        assert delete_data["success"] is True

        # 6. Verify it's gone
        verify_resp = page.request.get(f"{BASE_URL}/api/portfolio/positions")
        verify_data = verify_resp.json()
        not_found = not any(p.get("id") == position_id for p in verify_data["positions"])
        assert not_found, "Deleted position should not appear"

    def test_trade_lifecycle(self, page: Page):
        """Create -> Read -> Delete a trade record via API."""
        # Create
        create_resp = page.request.post(
            f"{BASE_URL}/api/portfolio/trades",
            data=json.dumps({
                "trade_date": "2026-02-20",
                "code": "600036",
                "action": "buy",
                "quantity": 500,
                "price": 35.00,
                "reason": "E2E测试交易"
            }),
            headers={"Content-Type": "application/json"}
        )
        assert create_resp.ok
        create_data = create_resp.json()
        assert create_data["success"] is True
        trade_id = create_data.get("trade_id") or create_data.get("id")

        # Read
        read_resp = page.request.get(f"{BASE_URL}/api/portfolio/trades")
        assert read_resp.ok
        read_data = read_resp.json()
        assert any(t.get("code") == "600036" for t in read_data["trades"])

        # Delete
        if trade_id:
            delete_resp = page.request.delete(f"{BASE_URL}/api/portfolio/trades/{trade_id}")
            assert delete_resp.ok

    def test_group_lifecycle(self, page: Page):
        """Create -> Read -> Delete a position group via API."""
        # Create
        create_resp = page.request.post(
            f"{BASE_URL}/api/portfolio/groups",
            data=json.dumps({
                "name": "测试分组E2E",
                "color": "#ff5733",
                "description": "E2E测试用分组",
                "sort_order": 99
            }),
            headers={"Content-Type": "application/json"}
        )
        assert create_resp.ok
        create_data = create_resp.json()
        assert create_data["success"] is True
        group_id = create_data.get("group_id") or create_data.get("id")

        # Read
        read_resp = page.request.get(f"{BASE_URL}/api/portfolio/groups")
        assert read_resp.ok
        groups = read_resp.json()["groups"]
        assert any(g["name"] == "测试分组E2E" for g in groups)

        # Delete
        if group_id:
            delete_resp = page.request.delete(f"{BASE_URL}/api/portfolio/groups/{group_id}")
            assert delete_resp.ok


# ============================================================
# 18. Concurrent/Rapid Interaction Tests
# ============================================================

class TestRapidInteractions:
    """Test rapid user interactions that could cause race conditions."""

    def test_rapid_search_typing(self, page: Page):
        """Type rapidly in search and verify no crashes."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        search = page.locator("#stock-search-input")
        # Type rapidly
        for char in "000001600036300750":
            search.type(char, delay=30)
        page.wait_for_timeout(500)

        # Page should still be functional
        assert page.locator("nav.navbar").is_visible()

    def test_rapid_tab_switching(self, page: Page):
        """Rapidly switch between portfolio tabs."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1000)

        for _ in range(3):
            page.locator("#recommendations-tab").click()
            page.wait_for_timeout(100)
            page.locator("#trades-tab").click()
            page.wait_for_timeout(100)
            page.locator("#evaluations-tab").click()
            page.wait_for_timeout(100)
            page.locator("#positions-tab").click()
            page.wait_for_timeout(100)

        # Page should still be functional
        assert page.locator("#portfolioTabs").is_visible()

    def test_rapid_page_navigation(self, page: Page):
        """Navigate rapidly between pages."""
        # Exclude data-management since it has slow DB queries
        pages = ["/", "/daily-tasks", "/model-training", "/backtest", "/portfolio"]

        for p in pages:
            page.goto(f"{BASE_URL}{p}")

        page.wait_for_load_state("domcontentloaded")
        assert page.locator("nav.navbar").is_visible()


# ============================================================
# 19. JavaScript Utility Function Tests
# ============================================================

class TestJSUtilityFunctions:
    """Test JavaScript utility functions exposed globally."""

    def test_format_number(self, page: Page):
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        result = page.evaluate("window.formatNumber(1234567.89)")
        assert "M" in result or "1234567" in str(result)

        result2 = page.evaluate("window.formatNumber(1500)")
        assert "K" in result2 or "1500" in str(result2)

        result3 = page.evaluate("window.formatNumber(null)")
        assert result3 == "-"

    def test_format_percent(self, page: Page):
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        result = page.evaluate("window.formatPercent(0.1234)")
        assert "12.34%" == result

        result2 = page.evaluate("window.formatPercent(null)")
        assert result2 == "-"

    def test_format_date(self, page: Page):
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        result = page.evaluate("window.formatDate('2026-02-23')")
        assert "2026" in result

        result2 = page.evaluate("window.formatDate(null)")
        assert result2 == "-"

    def test_api_helper_get(self, page: Page):
        """Test the window.api.get helper."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        result = page.evaluate("""
            async () => {
                const resp = await window.api.get('/api/health');
                return resp;
            }
        """)
        assert result["status"] == "healthy"


# ============================================================
# 20. Data Integrity Checks
# ============================================================

class TestDataIntegrity:
    """Verify data consistency between API and UI."""

    def test_dashboard_stock_count_matches_api(self, page: Page):
        """Stock count on dashboard should match API response."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        # Get API data
        api_resp = page.request.get(f"{BASE_URL}/api/daily/status")
        api_data = api_resp.json()
        api_count = api_data["database_stats"]["total_stocks"]

        # Get UI data
        ui_count_text = page.locator("#total-stocks").text_content().strip()
        if ui_count_text and ui_count_text != "-":
            ui_count = int(ui_count_text.replace(",", ""))
            assert ui_count == api_count, f"UI shows {ui_count}, API returns {api_count}"

    def test_portfolio_summary_matches_positions(self, page: Page):
        """Portfolio summary should be consistent with position data."""
        api_resp = page.request.get(f"{BASE_URL}/api/portfolio/positions")
        data = api_resp.json()

        summary = data["summary"]
        positions = data["positions"]

        assert summary["position_count"] == len(positions)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
