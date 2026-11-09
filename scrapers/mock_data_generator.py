"""
High-fidelity synthetic data generator for HKEX & KRX social sentiment and price bars.
Generates realistic 10-day historical time-series with authentic Cantonese,
Chinese, and English retail commentary across Telegram, LIHKG, and Futu.
Ensures the pipeline is fully testable and reproducible out-of-the-box.
"""

from datetime import datetime, timedelta, timezone
import math
import random
from typing import Any, Dict, List, Tuple
from scrapers.base_scraper import SocialPost
import config

# Authentic sample messages by ticker and platform
MOCK_TEMPLATES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "0700.HK": {
        "lihkg": [
            {"text": "騰訊0700業績大好！AI雲端同手遊業務增長超預期，今次真係突破420阻力位，睇好衝上450！", "sentiment": 0.85},
            {"text": "0700呢個位有大戶沽空盤壓住，日日陰跌，穿埋410就考慮止蝕割肉啦，睇淡。", "sentiment": -0.65},
            {"text": "騰訊今日又回購10億，企鵝真係港股定海神針，長線投資首選，逢低吸納！", "sentiment": 0.75},
            {"text": "恒指今日跟住美股回調，0700橫盤震盪，無咩方向，等下星期議息結果先再部署。", "sentiment": 0.05},
            {"text": "騰訊新Game流水爆升，微信小店GMV翻倍，北水持續買入，目標價睇460！", "sentiment": 0.90},
            {"text": "睇住0700日線圖，MACD頂背馳，小心假突破回調去400蚊整固。", "sentiment": -0.45},
            {"text": "南向資金今日淨流入騰訊超過8億港元，大行調高評級至跑贏大市。", "sentiment": 0.70},
        ],
        "telegram": [
            {"text": "ALERT: $0700 Tencent breaks key resistance at 418 HKD on 1.4x volume. Momentum turning strongly bullish. Next target 435.", "sentiment": 0.80},
            {"text": "Tencent daily buyback update: Repurchased 2.45M shares for HKD 1.02B. Consistent floor support around 412 HKD.", "sentiment": 0.60},
            {"text": "Unusual put option flow detected on 0700.HK Nov 400 strike. Retail caution advised ahead of macro CPI print.", "sentiment": -0.40},
            {"text": "Tech sector update: Tencent Cloud launches new enterprise deep-learning API stack. Enterprise cloud ARR up 24% YoY.", "sentiment": 0.70},
        ],
        "futu_moomoo": [
            {"text": "牛軍集合！騰訊這波升浪非常紮實，換手充分，機構席位大買，坐穩扶好！", "sentiment": 0.80},
            {"text": "415壓力位磨了好幾天，量能沒跟上，短期可能回踩408支撐位確認。", "sentiment": -0.30},
            {"text": "剛看完大摩研報，將騰訊目標價調升至480，基本面非常硬淨！", "sentiment": 0.75},
        ],
    },
    "9988.HK": {
        "lihkg": [
            {"text": "9988阿里真係廢柴，年年話反彈，次次彈少少又破底，85蚊都守唔住，睇淡到極點！", "sentiment": -0.85},
            {"text": "阿里雲入選政府重點採購名單，淘寶雙11成交額創歷史新高，9988終於見底回升！", "sentiment": 0.80},
            {"text": "BABA美股盤前跌2%，聽日9988又要大陰燭開市，千祈咪抄底！", "sentiment": -0.70},
            {"text": "9988而家市盈率得8倍，現金流多到數唔曬，低估值修復只是時間問題，買入等收成。", "sentiment": 0.65},
            {"text": "電商競爭太激烈，拼多多同抖音夾擊，阿里核心淘寶利潤率持續承壓。", "sentiment": -0.55},
        ],
        "telegram": [
            {"text": "$9988 Alibaba rebound attempts rejected at 90.50 resistance. Heavy institutional selling visible in dark pool.", "sentiment": -0.60},
            {"text": "Alibaba initiates aggressive share repurchase: $4.1B repurchased in Q3. Bullish floor solidifying near 84 HKD.", "sentiment": 0.55},
            {"text": "Macro alert: China retail consumption vouchers spark tech rally. 9988 up 3.2% in active morning trading.", "sentiment": 0.65},
        ],
        "futu_moomoo": [
            {"text": "阿里終於有起色，今日突破20天線，MACD金叉，看好補回前高缺口！", "sentiment": 0.70},
            {"text": "拿了兩年9988終於微利解套，先出本金，留利潤繼續觀察，不貪。", "sentiment": 0.10},
        ],
    },
    "3690.HK": {
        "lihkg": [
            {"text": "美團3690沙特KeeTa外賣業務市佔第一！海外擴張神速，中東成新增長引擎，爆升！", "sentiment": 0.90},
            {"text": "內地消費降級，美團客單價下跌，核心本地商業利潤被侵蝕，睇淡3690。", "sentiment": -0.60},
            {"text": "3690今日升穿180阻力，成交急增，沽空比率降至8%，明顯有大戶夾淡倉！", "sentiment": 0.85},
            {"text": "美團閃購同無人機配送業務持續放量，高盛重申買入評級，目標價215。", "sentiment": 0.75},
        ],
        "telegram": [
            {"text": "$3690 Meituan volume breakout above 175 HKD. Bullish sentiment surge across Hong Kong retail forums.", "sentiment": 0.80},
            {"text": "Meituan expansion update: Middle East operations achieving unit economics faster than projected.", "sentiment": 0.70},
            {"text": "Short squeeze underway on 3690.HK as borrow fees spike to 3.8%. Target 190 HKD.", "sentiment": 0.75},
        ],
        "futu_moomoo": [
            {"text": "美團基本面真的強，外賣護城河極深，每次回調都是給機會上車！", "sentiment": 0.75},
            {"text": "今天衝高182受阻回落，短線做T的高拋低吸，中線繼續看好。", "sentiment": 0.20},
        ],
    },
    "0005.HK": {
        "lihkg": [
            {"text": "大笨象0005滙豐又派高息，特別息加季度息年息率接近8厘，收息黨最愛，買入！", "sentiment": 0.85},
            {"text": "美聯儲如果減息周期展開，滙豐淨息差NIM見頂，明年盈利增長動力放緩，睇淡。", "sentiment": -0.50},
            {"text": "滙豐70蚊大關企得好穩，英國及亞洲業務重組節省成本，大笨象都會跳舞！", "sentiment": 0.70},
            {"text": "大笨象防守力一流，恒指大跌佢跌最少，資金避險首選！", "sentiment": 0.65},
        ],
        "telegram": [
            {"text": "$0005 HSBC holding steady above 71.0 HKD. Institutional dividend reinvestment inflows keeping bid strong.", "sentiment": 0.60},
            {"text": "HSBC announces $3B share buyback alongside solid Q3 CET1 capital ratio of 15.2%.", "sentiment": 0.75},
            {"text": "FX & interest rate sensitivities: HSBC guidance confirms resilient net interest margin through 2026.", "sentiment": 0.50},
        ],
        "futu_moomoo": [
            {"text": "滙豐就是穩，退休收息利器，每季派息準時入賬，長線繼續持有。", "sentiment": 0.70},
            {"text": "高位72.5有些獲利回吐盤，但下檔買盤依然厚實。", "sentiment": 0.25},
        ],
    },
    "1211.HK": {
        "lihkg": [
            {"text": "比亞迪1211月銷量突破50萬架！打破全球新能源車歷史紀錄，國產電車之王，睇好300！", "sentiment": 0.95},
            {"text": "歐盟加徵關稅阻礙出海，各國貿易保護主義抬頭，比亞迪估值受壓，睇淡。", "sentiment": -0.55},
            {"text": "第五代DM混動技術太震撼，滿油滿電2100公里，合資燃油車完全無得打，買入1211！", "sentiment": 0.90},
            {"text": "比亞迪今日高位回吐，280阻力較大，等回踩270再加倉。", "sentiment": 0.10},
        ],
        "telegram": [
            {"text": "$1211 BYD breaks all-time monthly delivery record. Exports surging across Southeast Asia & LatAm.", "sentiment": 0.90},
            {"text": "BYD smart driving upgrade deployment rolling out to over 1M vehicles. AI tech integration accelerating.", "sentiment": 0.75},
            {"text": "Analyst note: BYD gross margins expand to 22.4% despite fierce domestic price competition.", "sentiment": 0.80},
        ],
        "futu_moomoo": [
            {"text": "比亞迪這業績簡直逆天，海外建廠進展順利，長線目標看400！", "sentiment": 0.88},
            {"text": "新能源板塊今天整體調整，1211抗跌性很強，量價結構健康。", "sentiment": 0.40},
        ],
    },
    "005930.KS": {
        "naver_finance": [
            {"text": "삼성전자 8만전자 가즈아! 엔비디아 HBM3E 공급 승인 임박 소식 나왔다, 전량 매수 추천!", "sentiment": 0.90},
            {"text": "삼전 외인 3000억 순매도... 55000원 지지선 깨지면 투매 나온다 조심해라", "sentiment": -0.75},
            {"text": "파운드리 적자 축소되고 D램 고정가격 상승세 지속, 바닥 다지고 반등 시작한다", "sentiment": 0.70},
            {"text": "개미들 다 털리고 나니까 반등하네. 6만원 안착하면 65000원까지 열려있다", "sentiment": 0.65},
        ],
        "telegram": [
            {"text": "$005930 Samsung Electronics qualifies 12-layer HBM3E memory for hyperscaler deployment. Buy order flow surge.", "sentiment": 0.85},
            {"text": "KRX Semiconductor Sector: Foreign net outflow slows down; Samsung RSI at 34 signals oversold technical rebound.", "sentiment": 0.60},
            {"text": "Samsung foundry fab yield updates indicate steady progress on 2nm GAA process nodes.", "sentiment": 0.70},
        ],
    },
    "000660.KS": {
        "naver_finance": [
            {"text": "SK하이닉스 HBM 독점 체제 굳건! 4분기 사상 최대 영업이익 전망, 20만닉스 간다", "sentiment": 0.95},
            {"text": "고점 부근에서 단기 차익실현 매물 출회 중. 175000원 지지 확인하고 들어가자", "sentiment": -0.30},
            {"text": "외인 기관 양매수 유입! 메모리 슈퍼사이클 최대 수혜주는 역시 하닉이다", "sentiment": 0.85},
        ],
        "telegram": [
            {"text": "$000660 SK Hynix begins volume production of HBM4 ahead of schedule. Institutional targets revised to 220,000 KRW.", "sentiment": 0.85},
            {"text": "SK Hynix operating margins lead global peer group in high-bandwidth memory tier.", "sentiment": 0.80},
        ],
    },
    "035420.KS": {
        "naver_finance": [
            {"text": "네이버 웹툰 나스닥 상장 효과와 AI 검색 광고 수익화 가속! 20만원 돌파 기대", "sentiment": 0.80},
            {"text": "내수 광고 시장 부진으로 밸류에이션 리레이팅 지연, 박스권 횡보 예상", "sentiment": -0.40},
            {"text": "클라우드 부문 엔터프라이즈 AI 수주 증가세 확인. 실적 턴어라운드 진입", "sentiment": 0.75},
        ],
        "telegram": [
            {"text": "$035420 NAVER launches corporate AI hyper-clova suite with high-margin recurring ARR.", "sentiment": 0.75},
        ],
    },
    "005380.KS": {
        "naver_finance": [
            {"text": "현대차 인도 법인 상장 대박! 주주환원 자사주 매입 소각 공시, 배당수익률 6% 매수", "sentiment": 0.90},
            {"text": "미국 관세 리스크 부각으로 단기 조정 국면, 저가 분할매수 기회", "sentiment": 0.45},
            {"text": "하이브리드 판매 비중 급증으로 수익성 방어 성공, 밸류에이션 매력 충분", "sentiment": 0.75},
        ],
        "telegram": [
            {"text": "$005380 Hyundai Motor hybrid deliveries surge 42% YoY in North America, beating margin consensus.", "sentiment": 0.80},
        ],
    },
    "035720.KS": {
        "naver_finance": [
            {"text": "카카오 신임 경영진 쇄신안 발표, 비핵심 자산 매각으로 수익성 대폭 개선 전망", "sentiment": 0.70},
            {"text": "사법 리스크 여전히 발목 잡고 있음. 거래대금 줄어들고 추가 하락 경계", "sentiment": -0.65},
            {"text": "카카오톡 탭 개편으로 일간 활성 사용자수 반등세 확인", "sentiment": 0.55},
        ],
        "telegram": [
            {"text": "$035720 Kakao restructuring underway; core messenger fintech GMV stable.", "sentiment": 0.50},
        ],
    },
}


def generate_mock_social_posts(
    ticker: str,
    days: int = 10,
    posts_per_day: int = 12,
    seed: int = 42,
) -> List[SocialPost]:
    """
    Generate synthetic SocialPost objects for a ticker across Telegram, LIHKG, and Futu
    for the exact past `days` window, aligned strictly to UTC.
    """
    random.seed(seed + sum(ord(c) for c in ticker))
    now = datetime.now(timezone.utc)
    posts: List[SocialPost] = []

    templates_by_source = MOCK_TEMPLATES.get(ticker, MOCK_TEMPLATES["0700.HK"])

    for d in range(days - 1, -1, -1):
        day_date = now - timedelta(days=d)
        # Create day-level sentiment drift
        day_bias = math.sin(d * 0.8) * 0.25

        # Variable post counts to simulate buzz anomalies
        count = int(posts_per_day * random.uniform(0.6, 1.8))
        if d in (2, 7):  # Buzz spike days
            count = int(count * 2.2)

        for i in range(count):
            # Timestamp uniformly spread throughout the trading & evening day
            post_time = day_date.replace(
                hour=random.randint(1, 15),
                minute=random.randint(0, 59),
                second=random.randint(0, 59),
                microsecond=0,
            )

            # Pick source according to exchange
            is_krx = ticker.endswith(".KS") or ticker in config.KRX_ASSETS
            if is_krx:
                source = random.choices(["naver_finance", "telegram"], weights=[0.60, 0.40])[0]
                source_templates = templates_by_source.get(source, templates_by_source.get("naver_finance", []))
            else:
                source = random.choices(["telegram", "lihkg", "futu_moomoo"], weights=[0.35, 0.40, 0.25])[0]
                source_templates = templates_by_source.get(source, templates_by_source.get("lihkg", []))

            if not source_templates:
                continue

            template = random.choice(source_templates)

            base_sentiment = template["sentiment"] + day_bias + random.gauss(0, 0.1)
            norm_sentiment = max(-1.0, min(1.0, base_sentiment))

            # Engagement metrics realistic for each platform
            if source == "telegram":
                if is_krx:
                    author = random.choice(["@krx_traders", "@yeouido_insider", "@korea_value_up", "@seoul_quant"])
                else:
                    author = random.choice(["@hkstockgains", "@canton_trader", "@alpha_hangseng", "@hongkong_equity_flow", "@hsi_alpha"])
                followers = random.randint(1500, 48000)
                views = int(followers * random.uniform(0.15, 0.65))
                likes = int(views * random.uniform(0.01, 0.08))
                reposts = int(views * random.uniform(0.005, 0.03))
            elif source == "naver_finance":
                followers = None  # Naver discussion boards do not expose follower counts
                views = random.randint(400, 9500)
                likes = random.randint(15, 220)
                reposts = random.randint(3, 85)
                author = f"NaverUser_{random.randint(1000, 9999)}"
            elif source == "lihkg":
                followers = None  # LIHKG does not expose user follower counts
                views = None      # LIHKG thread lists do not expose view counts (replies & likes only)
                likes = random.randint(5, 280)
                reposts = random.randint(2, 95)  # Replies in forum context
                author = f"LIHKG_User_{random.randint(10000, 99999)}"
            else:  # Futu / Moomoo
                followers = random.randint(200, 18000)
                views = random.randint(500, 8500)
                likes = random.randint(10, 190)
                reposts = random.randint(1, 40)
                author = f"Futu_Investor_{random.randint(100, 999)}"

            label = "BULLISH" if norm_sentiment > 0.15 else ("BEARISH" if norm_sentiment < -0.15 else "NEUTRAL")

            post = SocialPost(
                post_id=f"mock_{source}_{ticker}_{d}_{i}_{random.randint(1000, 9999)}",
                source=source,
                ticker=ticker,
                timestamp_utc=post_time.isoformat(),
                text=f"[MOCK DEMO] {template['text']}",
                author=author,
                author_followers=followers,
                likes=likes,
                reposts=reposts,
                views=views,
                url=f"https://{source}.example.com/post/{d}_{i}",
                sentiment_score=round(norm_sentiment, 4),
                sentiment_label=label,
                is_mock=True,
                raw_metadata={"day_offset": d, "platform": source, "is_synthetic": True},
            )
            posts.append(post)

    # Sort strictly chronologically by UTC timestamp
    posts.sort(key=lambda p: p.timestamp_utc)
    return posts
