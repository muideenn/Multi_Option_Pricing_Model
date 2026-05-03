#!/usr/bin/env python
# coding: utf-8

# # Multi-Company Options Pricing Dashboard
# ### Binomial · Black-Scholes · GARCH(1,1) · Cross-Company Analysis
# 
# **Author:** Muhideen Ogunlowo
# 
# **Companies:** DIS · AAPL · MSFT · GOOGL · AMZN · TSLA · NVDA · META · NFLX · JPM
# 
# Run all cells → open **http://127.0.0.1:8049**
# 

# In[1]:


import subprocess, sys
pkgs = ['dash','dash-bootstrap-components','plotly','yfinance','pandas','numpy','scipy','arch']
subprocess.check_call([sys.executable, '-m', 'pip', 'install', *pkgs, '-q'])
print('Packages ready.')


# In[2]:


import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model
import yfinance as yf
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
print('Imports OK')


# In[3]:


# ── Company registry ─────────────────────────────────────────────
COMPANIES = {
    'DIS':  dict(name='Walt Disney Company',   sector='Communications', mu=0.0001, sig=0.014, start=100),
    'AAPL': dict(name='Apple Inc.',             sector='Technology',     mu=0.0005, sig=0.016, start=180),
    'MSFT': dict(name='Microsoft Corporation',  sector='Technology',     mu=0.0006, sig=0.015, start=380),
    'GOOGL':dict(name='Alphabet Inc.',          sector='Technology',     mu=0.0004, sig=0.017, start=170),
    'AMZN': dict(name='Amazon.com Inc.',        sector='Consumer Disc.', mu=0.0003, sig=0.019, start=190),
    'TSLA': dict(name='Tesla Inc.',             sector='Consumer Disc.', mu=0.0002, sig=0.032, start=200),
    'NVDA': dict(name='NVIDIA Corporation',     sector='Technology',     mu=0.0010, sig=0.028, start=800),
    'META': dict(name='Meta Platforms Inc.',    sector='Technology',     mu=0.0005, sig=0.023, start=500),
    'NFLX': dict(name='Netflix Inc.',           sector='Communications', mu=0.0003, sig=0.021, start=600),
    'JPM':  dict(name='JPMorgan Chase & Co.',   sector='Financials',     mu=0.0003, sig=0.013, start=200),
}

def _mock_data(ticker):
    c = COMPANIES.get(ticker, COMPANIES['DIS'])
    np.random.seed(abs(hash(ticker)) % (2**31))
    rng = pd.bdate_range(end=pd.Timestamp.now(), periods=5*252)
    lr  = np.random.normal(c['mu'], c['sig'], len(rng))
    # Inject a brief stress period 2 years ago
    mid = len(rng) - 2*252
    lr[mid:mid+20] *= 3.0
    px  = np.exp(np.cumsum(lr)) * c['start']
    df  = pd.DataFrame({'Close': px, 'Log_Return': lr}, index=rng)
    for w in (21, 63, 126):
        df[f'HV_{w}'] = df['Log_Return'].rolling(w).std() * np.sqrt(252)
    return df.dropna()

def fetch_data(ticker='DIS', period='5y'):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty: raise ValueError('empty')
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        for w in (21, 63, 126):
            df[f'HV_{w}'] = df['Log_Return'].rolling(w).std() * np.sqrt(252)
        return df.dropna()
    except Exception as e:
        print(f'yfinance unavailable ({e}) -- synthetic data for {ticker}')
        return _mock_data(ticker)

# ── Pricing models ───────────────────────────────────────────────
def binomial_price(S,K,T,r,sigma,N=200,option_type='call',exercise='european'):
    if T<1e-9 or sigma<1e-9: return max(S-K,0) if option_type=='call' else max(K-S,0)
    dt=T/N; u=np.exp(sigma*np.sqrt(dt)); d=1/u
    p=(np.exp(r*dt)-d)/(u-d); disc=np.exp(-r*dt)
    j=np.arange(N+1); ST=S*u**(N-j)*d**j
    V=np.maximum(ST-K,0) if option_type=='call' else np.maximum(K-ST,0)
    for i in range(N-1,-1,-1):
        V=disc*(p*V[:-1]+(1-p)*V[1:])
        if exercise=='american':
            ji=np.arange(i+1); Si=S*u**(i-ji)*d**ji
            pf=np.maximum(Si-K,0) if option_type=='call' else np.maximum(K-Si,0)
            V=np.maximum(V,pf)
    return float(V[0])

def bs_price(S,K,T,r,sigma,option_type='call'):
    if T<1e-9 or sigma<1e-9: return max(S-K,0) if option_type=='call' else max(K-S,0)
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2) if option_type=='call' else K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def bs_greeks(S,K,T,r,sigma,option_type='call'):
    if T<1e-9 or sigma<1e-9: return dict(delta=0.,gamma=0.,theta=0.,vega=0.,rho=0.)
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T)); d2=d1-sigma*np.sqrt(T)
    pdf1=norm.pdf(d1)
    delta=norm.cdf(d1) if option_type=='call' else norm.cdf(d1)-1
    gamma=pdf1/(S*sigma*np.sqrt(T))
    base=-S*pdf1*sigma/(2*np.sqrt(T))
    theta=(base-r*K*np.exp(-r*T)*norm.cdf(d2))/365 if option_type=='call' else (base+r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    vega=S*pdf1*np.sqrt(T)/100
    rho=(K*T*np.exp(-r*T)*norm.cdf(d2) if option_type=='call' else -K*T*np.exp(-r*T)*norm.cdf(-d2))/100
    return dict(delta=delta,gamma=gamma,theta=theta,vega=vega,rho=rho)

def fit_garch(log_returns,p=1,q=1):
    model=arch_model(log_returns*100,vol='Garch',p=p,q=q,dist='normal')
    result=model.fit(disp='off')
    pm=result.params; keys=list(pm.index)
    omega=float(pm[next(k for k in keys if 'omega' in k.lower())])
    alpha=float(pm[next(k for k in keys if 'alpha' in k.lower())])
    beta =float(pm[next(k for k in keys if 'beta'  in k.lower())])
    return result, dict(omega=omega, alpha=alpha, beta=beta)

def garch_forecast(result, horizon=60):
    fc=result.forecast(horizon=horizon)
    return np.sqrt(fc.variance.values[-1,:])/100*np.sqrt(252)

def implied_vol(mkt_price,S,K,T,r,option_type='call',tol=1e-7):
    lo,hi=1e-6,6.0
    for _ in range(150):
        mid=(lo+hi)/2
        lo,hi=(mid,hi) if bs_price(S,K,T,r,mid,option_type)<mkt_price else (lo,mid)
        if hi-lo<tol: break
    return float(mid)

print('Models defined')


# In[4]:


C = dict(
    bg='#0d1117', surface='#161b22', surface2='#21262d', border='#30363d',
    text='#e6edf3', muted='#8b949e',
    teal='#39d0a0', purple='#a78bfa', amber='#fbbf24',
    coral='#fb7185', blue='#60a5fa', green='#4ade80',
    orange='#fb923c', pink='#f472b6',
)
MONO = "'JetBrains Mono','Fira Code',monospace"

# Per-company accent colours (cycles through palette)
TICKER_COLORS = {
    'DIS': C['teal'],  'AAPL': C['blue'],  'MSFT': C['purple'],
    'GOOGL':C['amber'],'AMZN': C['orange'],'TSLA': C['coral'],
    'NVDA': C['green'],'META': C['pink'],   'NFLX': C['coral'],
    'JPM':  C['blue'],
}

BASE_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=C['text'],family='Inter,sans-serif',size=12),
    colorway=[C['teal'],C['purple'],C['amber'],C['coral'],C['blue'],
              C['green'],C['orange'],C['pink']],
    xaxis=dict(gridcolor='#21262d',linecolor=C['border'],zerolinecolor='#21262d'),
    yaxis=dict(gridcolor='#21262d',linecolor=C['border'],zerolinecolor='#21262d'),
    legend=dict(bgcolor='rgba(0,0,0,0)',bordercolor=C['border'],borderwidth=1,font=dict(size=11)),
    margin=dict(l=48,r=20,t=44,b=36),
    hoverlabel=dict(bgcolor=C['surface2'],bordercolor=C['border'],font=dict(color=C['text'],family='Inter')),
)
CARD  = {'background':C['surface'], 'border':f"1px solid {C['border']}", 'borderRadius':'12px','padding':'14px 18px'}
CARD2 = {'background':C['surface2'],'border':f"1px solid {C['border']}", 'borderRadius':'8px', 'padding':'12px 16px'}
LBL   = {'color':C['muted'],'fontSize':'10px','fontWeight':'600','letterSpacing':'0.09em','textTransform':'uppercase','marginBottom':'4px'}
VAL   = {'fontSize':'20px','fontWeight':'600','fontFamily':MONO,'lineHeight':'1.15'}

def apply_layout(fig,title='',height=300,legend_h=False,**kw):
    lo=dict(**BASE_LAYOUT,height=height,title=dict(text=title,font=dict(size=13,color=C['muted']),x=0,xanchor='left'))
    if legend_h: lo['legend'].update(orientation='h',y=1.12,x=0)
    lo.update(kw); fig.update_layout(**lo); return fig

def metric(label,value,color,width=2):
    return dbc.Col(html.Div([html.Div(label,style=LBL),html.Div(value,style=VAL|{'color':color})],style=CARD),width=width,className='mb-2')

def slide_intro(phase_num,accent,title,body_lines,insight):
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div(f'Phase {phase_num}',style={'color':accent,'fontSize':'11px','fontWeight':'700',
                    'letterSpacing':'0.12em','textTransform':'uppercase','marginBottom':'6px'}),
                html.Div(title,style={'color':C['text'],'fontSize':'19px','fontWeight':'600','marginBottom':'10px'}),
                *[html.P(ln,style={'color':C['muted'],'fontSize':'13px','lineHeight':'1.7','margin':'0 0 5px 0'}) for ln in body_lines],
            ],width=8),
            dbc.Col(html.Div([
                html.Div('Key insight',style={'color':accent,'fontSize':'10px','fontWeight':'700',
                    'letterSpacing':'0.1em','textTransform':'uppercase','marginBottom':'8px'}),
                html.Div(insight,style={'color':C['text'],'fontSize':'13px','lineHeight':'1.65'}),
            ],style={'background':f"{accent}12",'border':f'1px solid {accent}40',
                'borderLeft':f'3px solid {accent}','borderRadius':'8px','padding':'14px 16px','height':'100%'}),
            width=4),
        ],className='g-3'),
    ],style={'background':C['surface'],'border':f"1px solid {C['border']}",
        'borderRadius':'12px','padding':'20px 24px','marginBottom':'20px'})

def chart_note(*cols):
    return html.Div(dbc.Row([dbc.Col(html.Div(txt,style={'color':C['muted'],'fontSize':'12px','lineHeight':'1.7'}),width=w)
        for txt,w in cols],className='g-3'),style=CARD2|{'marginTop':'14px'})

print('Style system ready')


# In[5]:


# ══ Tab 1: Market Data ═══════════════════════════════════════════
def render_market(df, S, hv21, ticker, acc):
    cname = COMPANIES[ticker]['name']
    intro = slide_intro(1, acc,
        f'Market Data Analysis — {cname} ({ticker})',
        [
            f'Five years of daily closing prices for {ticker} are sourced and analysed. '
            'Log returns r_t = ln(P_t / P_{t-1}) are computed for stationarity and used throughout '
            'all subsequent pricing models as the primary volatility input.',
            'Three rolling historical volatility windows are estimated — 21-day (one month), '
            '63-day (one quarter), and 126-day (half year) — and annualised by multiplying '
            'by sqrt(252). The shortest window is most reactive; the longest is smoothest.',
        ],
        'Heavy tails and negative skew are present in most equity return distributions. '
        'Kurtosis above 3 means extreme moves are more frequent than a normal distribution '
        'predicts — directly motivating GARCH modelling to capture volatility clustering.'
    )
    fig_p=go.Figure()
    fig_p.add_trace(go.Scatter(x=df.index,y=df['Close'],mode='lines',name=ticker,
        line=dict(color=acc,width=1.5),fill='tozeroy',fillcolor=f'{acc}0d'))
    apply_layout(fig_p,f'{ticker} — Closing Price (5Y)',height=265)
    fig_p.update_xaxes(title_text='Date'); fig_p.update_yaxes(title_text='Price ($)')

    ret=df['Log_Return'].dropna(); mu,sd=float(ret.mean()),float(ret.std())
    x_fit=np.linspace(float(ret.min()),float(ret.max()),300)
    bw=(float(ret.max())-float(ret.min()))/80
    y_fit=norm.pdf(x_fit,mu,sd)*len(ret)*bw
    fig_r=go.Figure()
    fig_r.add_trace(go.Histogram(x=ret,nbinsx=80,name='Log returns',marker_color=acc,opacity=0.7))
    fig_r.add_trace(go.Scatter(x=x_fit,y=y_fit,mode='lines',
        line=dict(color=C['amber'],width=2),name='Normal fit'))
    apply_layout(fig_r,'Log-Return Distribution',height=265,legend_h=True)

    fig_v=go.Figure()
    for col,clr,lbl in [('HV_21',acc,'21d HV'),('HV_63',C['amber'],'63d HV'),('HV_126',C['muted'],'126d HV')]:
        fig_v.add_trace(go.Scatter(x=df.index,y=df[col]*100,mode='lines',name=lbl,
            line=dict(color=clr,width=1.5)))
    apply_layout(fig_v,'Annualised Historical Volatility (%)',height=265,legend_h=True)
    fig_v.update_yaxes(title_text='Vol (%)',ticksuffix='%')

    stats=[('Last Close',f'${S:.2f}',acc),
           ('52W High',  f'${df["Close"].max():.2f}',C['green']),
           ('52W Low',   f'${df["Close"].min():.2f}',C['coral']),
           ('HV 21d',    f'{hv21*100:.1f}%',C['amber']),
           ('Skewness',  f'{float(ret.skew()):.3f}',C['muted']),
           ('Excess Kurt',f'{float(ret.kurt()):.3f}',C['muted'])]
    stat_row=dbc.Row([metric(l,v,c,width=2) for l,v,c in stats],className='g-2 mb-3')

    return html.Div([intro, stat_row,
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_p,config={'displayModeBar':False}),width=12)],className='mb-2'),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_r,config={'displayModeBar':False}),width=6),
                 dbc.Col(dcc.Graph(figure=fig_v,config={'displayModeBar':False}),width=6)]),
        chart_note(
            ('Price chart: each company trades in a different regime — '
             'compare price levels and trend direction across selections.', 4),
            ('Return histogram: the amber normal curve shows how much fatter '
             'the actual tails are. High-vol names (TSLA, NVDA) show wider histograms.', 4),
            ('Rolling HV: shorter windows (21d) react fast to vol spikes; '
             'longer windows (126d) show the structural vol regime for each name.', 4),
        ),
    ])


# ══ Tab 2: Binomial ══════════════════════════════════════════════
def render_binomial(S, K, T, r, sigma, opt_type, exercise, ticker, acc):
    intro = slide_intro(2, acc,
        f'CRR Binomial Tree — {ticker} Option Pricing',
        [
            'The Cox-Ross-Rubinstein (1979) binomial model builds a recombining lattice. '
            'Up/down factors u = e^(σ√Δt) and d = 1/u are calibrated to the stock volatility. '
            'Risk-neutral probability p = (e^(rΔt) − d)/(u − d) ensures no-arbitrage pricing.',
            'American exercise is handled at each node by comparing hold value with immediate payoff. '
            'This is critical for put options — early exercise can be optimal when deep in-the-money '
            'because the interest on the strike exceeds remaining time value.',
        ],
        'Convergence to Black-Scholes is oscillatory (even/odd N alternate above/below). '
        'The error decays as O(1/N). N=200 gives sub-cent precision for standard maturities. '
        'For American options, no closed-form exists — the binomial tree is the benchmark.'
    )
    bs_val=bs_price(S,K,T,r,sigma,opt_type)
    N_vals=list(range(5,305,5))
    b_prices=[binomial_price(S,K,T,r,sigma,n,opt_type,exercise) for n in N_vals]
    errors=[abs(p-bs_val) for p in b_prices]

    fig_c=make_subplots(rows=2,cols=1,shared_xaxes=True,
        subplot_titles=['Binomial price vs N steps','|Error| vs Black-Scholes'],
        vertical_spacing=0.14)
    fig_c.add_trace(go.Scatter(x=N_vals,y=b_prices,mode='lines',name='Binomial',
        line=dict(color=acc,width=2)),row=1,col=1)
    fig_c.add_hline(y=bs_val,line_color=C['amber'],line_dash='dash',line_width=1.5,
        annotation_text=f'B-S ${bs_val:.3f}',annotation_font_color=C['amber'],row=1,col=1)
    fig_c.add_trace(go.Scatter(x=N_vals,y=errors,mode='lines',name='|Error|',
        line=dict(color=C['coral'],width=1.5),fill='tozeroy',fillcolor='rgba(251,113,133,0.1)'),row=2,col=1)
    fig_c.update_layout(**BASE_LAYOUT,height=400,showlegend=False,
        title=dict(text=f'CRR Convergence — {ticker}',font=dict(size=13,color=C['muted'])))
    fig_c.update_xaxes(title_text='N (Steps)',row=2,gridcolor='#21262d')
    fig_c.update_yaxes(title_text='Price ($)',row=1,gridcolor='#21262d')
    fig_c.update_yaxes(title_text='|Error| ($)',row=2,gridcolor='#21262d')

    strikes=np.linspace(max(S*0.6,1),S*1.4,60)
    fig_k=go.Figure()
    fig_k.add_trace(go.Scatter(x=strikes,
        y=[binomial_price(S,k,T,r,sigma,100,opt_type,exercise) for k in strikes],
        mode='lines',name='Binomial N=100',line=dict(color=acc,width=2)))
    fig_k.add_trace(go.Scatter(x=strikes,
        y=[bs_price(S,k,T,r,sigma,opt_type) for k in strikes],
        mode='lines',name='Black-Scholes',line=dict(color=C['amber'],width=2,dash='dash')))
    fig_k.add_vline(x=S,line_color=acc,line_dash='dot',annotation_text='Spot',annotation_font_color=acc)
    fig_k.add_vline(x=K,line_color=C['coral'],line_dash='dot',
        annotation_text='K',annotation_font_color=C['coral'],annotation_position='top right')
    apply_layout(fig_k,f'{opt_type.title()} Price vs Strike — {ticker}',height=290,legend_h=True)
    fig_k.update_xaxes(title_text='Strike ($)'); fig_k.update_yaxes(title_text='Price ($)')

    dt=T/100; u=np.exp(sigma*np.sqrt(dt)); d=1/u; p=(np.exp(r*dt)-d)/(u-d)
    snaps=[(n,binomial_price(S,K,T,r,sigma,n,opt_type,exercise)) for n in [10,50,100,200]]
    err200=abs(snaps[-1][1]-bs_val)
    snap_row=dbc.Row([
        *[metric(f'Binomial N={n}',f'${v:.4f}',acc,width=2) for n,v in snaps],
        metric('Black-Scholes',f'${bs_val:.4f}',C['amber'],width=2),
        metric('Error N=200',  f'${err200:.5f}',C['coral'],width=2),
    ],className='g-2 mb-3')
    params_box=html.Div([
        html.Div('CRR Parameters  (N=100)',style=LBL|{'marginBottom':'10px'}),
        dbc.Row([dbc.Col(html.Div([html.Span(k+' = ',style={'color':C['muted'],'fontFamily':MONO,'fontSize':'12px'}),
            html.Span(v,style={'color':acc,'fontFamily':MONO})]),width=3)
            for k,v in [('u',f'{u:.6f}'),('d',f'{d:.6f}'),('p*',f'{p:.6f}'),('Δt',f'{dt:.6f}')]]),
    ],style=CARD|{'marginBottom':'16px'})

    return html.Div([intro, snap_row, params_box,
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_c,config={'displayModeBar':False}),width=7),
                 dbc.Col(dcc.Graph(figure=fig_k,config={'displayModeBar':False}),width=5)]),
        chart_note(
            ('Convergence oscillates around B-S (dashed amber). Each N=5 step '
             'reduces the error roughly in half — visible in the lower error panel.', 6),
            ('The strike curve shows how both models agree closely for at-the-money '
             'options. Deep ITM/OTM deviations shrink as N increases.', 6),
        ),
    ])


# ══ Tab 3: Black-Scholes ═════════════════════════════════════════
def render_bs(S, K, T, r, sigma, opt_type, exercise, ticker, acc):
    intro = slide_intro(3, acc,
        f'Black-Scholes Model — {ticker} Greeks & Payoff',
        [
            'The Black-Scholes formula provides a closed-form solution under geometric Brownian motion '
            'with constant volatility. For a call: C = S·N(d1) − K·e^{−rT}·N(d2). '
            'The formula decomposes into the expected payoff (S·N(d1)) minus the '
            'discounted strike cost (K·e^{−rT}·N(d2)).',
            'The five Greeks measure sensitivity to each model input. Delta-hedging (Δ-hedging) '
            'uses Delta to maintain a neutral portfolio. Gamma and Vega are critical for '
            'volatility trading strategies. Theta quantifies the cost of holding options over time.',
        ],
        f'For {ticker}, the Delta surface shows how sensitivity shifts non-linearly across '
        'both spot and vol regimes. Near the money, Vega is highest — meaning vol changes '
        'matter most when the option is most uncertain about finishing in or out of the money.'
    )
    p_bs=bs_price(S,K,T,r,sigma,opt_type)
    p_bin=binomial_price(S,K,T,r,sigma,200,opt_type,exercise)
    g=bs_greeks(S,K,T,r,sigma,opt_type)

    spots=np.linspace(S*0.5,S*1.5,200)
    intr=np.maximum(spots-K,0) if opt_type=='call' else np.maximum(K-spots,0)
    bs_vals=np.array([bs_price(s,K,T,r,sigma,opt_type) for s in spots])
    fig_pay=go.Figure()
    fig_pay.add_trace(go.Scatter(x=spots,y=bs_vals,mode='lines',name='B-S Value',
        line=dict(color=acc,width=2.5)))
    fig_pay.add_trace(go.Scatter(x=spots,y=intr,mode='lines',name='Intrinsic',
        line=dict(color=C['muted'],width=1.5,dash='dash')))
    fig_pay.add_trace(go.Scatter(
        x=list(spots)+list(spots[::-1]),y=list(bs_vals)+list(intr[::-1]),
        fill='toself',fillcolor=f'{acc}10',line=dict(width=0),name='Time value',hoverinfo='skip'))
    fig_pay.add_vline(x=S,line_color=acc,line_dash='dot',
        annotation_text='Spot',annotation_font_color=acc)
    fig_pay.add_vline(x=K,line_color=C['coral'],line_dash='dot',
        annotation_text='Strike',annotation_font_color=C['coral'],annotation_position='top right')
    apply_layout(fig_pay,f'Option Value vs Spot — {ticker}',height=290,legend_h=True)
    fig_pay.update_xaxes(title_text='Spot ($)'); fig_pay.update_yaxes(title_text='Value ($)')

    sp_arr=np.linspace(S*0.7,S*1.3,40); vol_arr=np.linspace(0.08,0.80,40)
    SP,VL=np.meshgrid(sp_arr,vol_arr)
    D_surf=np.vectorize(lambda s,v: bs_greeks(s,K,T,r,v,opt_type)['delta'])(SP,VL)
    fig_d=go.Figure(go.Heatmap(x=sp_arr,y=vol_arr*100,z=D_surf,
        colorscale=[[0,'#0d2640'],[0.5,acc],[1,C['amber']]],
        colorbar=dict(title='Delta',tickfont=dict(color=C['text']),titlefont=dict(color=C['muted']))))
    fig_d.add_trace(go.Scatter(x=[S],y=[sigma*100],mode='markers',
        marker=dict(size=10,color=C['coral'],symbol='cross-thin',line=dict(width=2,color=C['coral'])),
        name='Current',showlegend=False))
    apply_layout(fig_d,f'Delta Surface — {ticker}',height=290)
    fig_d.update_xaxes(title_text='Spot ($)'); fig_d.update_yaxes(title_text='Vol (%)')

    greek_items=[
        ('Delta (Δ)', f'{g["delta"]:+.4f}', acc,
         '$1 stock move sensitivity. 0→1 for calls, −1→0 for puts.'),
        ('Gamma (Γ)', f'{g["gamma"]:.5f}', C['purple'],
         'Rate of change of Delta. Highest ATM near expiry.'),
        ('Theta (Θ)', f'${g["theta"]:+.4f}/d', C['amber'],
         'Daily time decay. Long options lose value each day.'),
        ('Vega (V)',  f'${g["vega"]:.4f}/%',  C['coral'],
         'Sensitivity to 1% vol change. Highest ATM.'),
        ('Rho (ρ)',   f'${g["rho"]:+.4f}/%',  C['blue'],
         'Sensitivity to 1% rate change. Smaller for short-dated.'),
    ]
    greek_cards=dbc.Row([
        dbc.Col(html.Div([html.Div(lbl,style=LBL),
            html.Div(val,style=VAL|{'color':clr,'fontSize':'18px','marginBottom':'5px'}),
            html.Div(desc,style={'color':C['muted'],'fontSize':'11px','lineHeight':'1.5'})],style=CARD),
            width=2,className='mb-2') for lbl,val,clr,desc in greek_items
    ]+[dbc.Col(html.Div([
        html.Div('B-S vs Binomial',style=LBL),
        html.Div([html.Span('B-S   ',style={'color':C['muted'],'fontFamily':MONO,'fontSize':'12px'}),
                  html.Span(f'${p_bs:.4f}',style={'color':acc,'fontFamily':MONO,'fontWeight':'600'})]),
        html.Div([html.Span('Binom ',style={'color':C['muted'],'fontFamily':MONO,'fontSize':'12px'}),
                  html.Span(f'${p_bin:.4f}',style={'color':C['purple'],'fontFamily':MONO,'fontWeight':'600'})]),
        html.Div([html.Span('Diff  ',style={'color':C['muted'],'fontFamily':MONO,'fontSize':'12px'}),
                  html.Span(f'${abs(p_bs-p_bin):.5f}',style={'color':C['coral'],'fontFamily':MONO,'fontWeight':'600'})]),
    ],style=CARD|{'lineHeight':'2.0'}),width=2,className='mb-2')],className='g-2 mb-3')

    return html.Div([intro, greek_cards,
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_pay,config={'displayModeBar':False}),width=6),
                 dbc.Col(dcc.Graph(figure=fig_d,  config={'displayModeBar':False}),width=6)]),
        chart_note(
            ('Payoff diagram: shaded area = time value. It is widest ATM and collapses '
             'to zero at expiry. Time value is what you pay beyond intrinsic.', 6),
            ('Delta surface: the red cross marks current spot and 21d HV. '
             'High-vol stocks (TSLA, NVDA) sit further up the y-axis, '
             'flattening the Delta gradient and making them harder to delta-hedge.', 6),
        ),
    ])


# ══ Tab 4: GARCH ═════════════════════════════════════════════════
def render_garch(df, S, K, T, r, opt_type, ticker, acc):
    intro = slide_intro(4, acc,
        f'GARCH(1,1) Volatility Forecast — {ticker}',
        [
            'Generalised Autoregressive Conditional Heteroskedasticity (GARCH) models time-varying '
            'variance: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}. α captures the impact of recent shocks; '
            'β captures persistence. The sum α+β < 1 ensures mean-reversion.',
            'We fit the model to log returns, generate a 60-day conditional variance forecast, '
            'annualise it, and substitute into Black-Scholes. This gives a regime-aware option price '
            'that responds to current market conditions rather than a static historical average.',
        ],
        'High α+β (close to 1) means volatility is persistent — a shock today influences '
        'vol for many days ahead. High-β stocks like TSLA or NVDA show very persistent vol, '
        'while lower-β financials like JPM mean-revert faster after earnings or macro events.'
    )
    returns=df['Log_Return'].dropna()
    garch_result,gparams=fit_garch(returns)
    cond_vol=garch_result.conditional_volatility/100*np.sqrt(252)
    fc_vol=garch_forecast(garch_result,60)
    fc_dates=pd.bdate_range(df.index[-1]+pd.Timedelta(days=1),periods=60)

    fig_gv=go.Figure()
    fig_gv.add_trace(go.Scatter(x=df.index[-252:],y=cond_vol[-252:]*100,mode='lines',
        name='GARCH Cond. Vol',line=dict(color=acc,width=1.5)))
    fig_gv.add_trace(go.Scatter(x=df.index[-252:],y=df['HV_21'][-252:]*100,mode='lines',
        name='21d HV',line=dict(color=C['muted'],width=1,dash='dot')))
    fig_gv.add_trace(go.Scatter(x=fc_dates,y=fc_vol*100,mode='lines',
        name='60d Forecast',line=dict(color=C['amber'],width=2.5)))
    band=fc_vol*0.18
    fig_gv.add_trace(go.Scatter(
        x=list(fc_dates)+list(fc_dates[::-1]),
        y=list((fc_vol+band)*100)+list((fc_vol-band)*100),
        fill='toself',fillcolor='rgba(251,191,36,0.08)',
        line=dict(width=0),name='95% band',hoverinfo='skip'))
    fig_gv.add_vline(x=str(df.index[-1]),line_color=C['muted'],line_dash='dash',line_width=1)
    apply_layout(fig_gv,f'GARCH(1,1) — {ticker} Conditional Vol + 60-Day Forecast',
        height=290,legend_h=True)
    fig_gv.update_yaxes(title_text='Annualised Vol (%)',ticksuffix='%')

    try:
        tkr_obj=yf.Ticker(ticker); exp_dates=tkr_obj.options
        exp_date=exp_dates[min(2,len(exp_dates)-1)]
        chain=getattr(tkr_obj.option_chain(exp_date),'calls' if opt_type=='call' else 'puts')
        chain=chain[(chain['bid']>0)&(chain['ask']>0)].copy()
        chain['mid']=(chain['bid']+chain['ask'])/2
        T_c=max((pd.to_datetime(exp_date)-pd.Timestamp.now()).days/365,0.01)
        gs=float(fc_vol[0])
        chain['garch_px']=chain['strike'].apply(lambda k: bs_price(S,k,T_c,r,gs,opt_type))
        chain['impl_vol']=chain.apply(
            lambda row: implied_vol(row['mid'],S,row['strike'],T_c,r,opt_type)*100
            if row['mid']>0.05 else np.nan,axis=1)
        chain=chain.dropna(subset=['impl_vol'])
        fig_iv=go.Figure()
        fig_iv.add_trace(go.Scatter(x=chain['strike'],y=chain['impl_vol'],mode='lines+markers',
            name='Impl. Vol',line=dict(color=acc,width=2),marker=dict(size=5)))
        fig_iv.add_hline(y=gs*100,line_color=C['amber'],line_dash='dash',
            annotation_text=f'GARCH: {gs*100:.1f}%',annotation_font_color=C['amber'])
        fig_iv.add_hline(y=float(df['HV_21'].iloc[-1])*100,line_color=C['teal'],line_dash='dot',
            annotation_text=f'21d HV',annotation_font_color=C['teal'])
        fig_iv.add_vline(x=S,line_color=C['muted'],line_dash='dot',annotation_text='Spot')
        apply_layout(fig_iv,f'Implied Vol Smile — {ticker} {exp_date}',height=290,legend_h=True)
        fig_iv.update_xaxes(title_text='Strike ($)'); fig_iv.update_yaxes(title_text='Impl. Vol (%)',ticksuffix='%')
        tbl_df=chain[['strike','mid','garch_px','impl_vol']].rename(columns={
            'strike':'Strike','mid':'Mkt Price','garch_px':'GARCH-BS','impl_vol':'Impl Vol %'}).copy()
        tbl_df['Diff']=( tbl_df['GARCH-BS']-tbl_df['Mkt Price']).round(4)
        tbl = dash_table.DataTable(
            data=tbl_df.head(14).round(3).to_dict('records'),
            columns=[{'name': c, 'id': c} for c in tbl_df.columns],
            style_table={'overflowX': 'auto', 'borderRadius': '8px', 'overflow': 'hidden'},
            style_cell={
            'backgroundColor': C['surface'],
            'color': C['text'],
            'border': f"1px solid {C['border']}",
            'fontFamily': "'JetBrains Mono',monospace",
            'fontSize': '11px',
            'padding': '7px 12px',
            'textAlign': 'right'
            },
            style_header={
            'backgroundColor': C['surface2'],
            'color': acc,
            'fontWeight': '600',
            'fontSize': '10px',
            'letterSpacing': '0.07em',
            'border': f"1px solid {C['border']}"
            },
            style_data_conditional=[  # pyright: ignore[reportArgumentType]
            {
                'if': {'filter_query': '{Diff} > 0', 'column_id': 'Diff'},
                'backgroundColor': f'{C["teal"]}22',
                'fontWeight': '600',
            },
            {
                'if': {'filter_query': '{Diff} < 0', 'column_id': 'Diff'},
                'backgroundColor': f'{C["coral"]}22',
                'fontWeight': '600',
            },
            ],
        )
        chain_sec=dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_iv,config={'displayModeBar':False}),width=7),
            dbc.Col(html.Div([html.Div('Market vs GARCH-BS',style=LBL|{'marginBottom':'8px'}),
                html.P('Teal = model cheap vs market. Coral = model expensive.',
                    style={'color':C['muted'],'fontSize':'11px','marginBottom':'8px'}),tbl],style=CARD),width=5),
        ],className='mt-3')
    except Exception as e:
        chain_sec=html.Div(f'Live options chain unavailable: {e}',
            style={'color':C['muted'],'padding':'12px','fontSize':'12px'})

    pers=gparams['alpha']+gparams['beta']
    gcard_row=dbc.Row([
        metric('omega',         f'{gparams["omega"]:.6f}',acc),
        metric('alpha (shock)', f'{gparams["alpha"]:.6f}',acc),
        metric('beta (persist)',f'{gparams["beta"]:.6f}', acc),
        metric('alpha+beta',    f'{pers:.6f}',             C['amber']),
        metric('GARCH Vol 1d',  f'{fc_vol[0]*100:.2f}%',  C['amber']),
        metric('Log-Likeli.',   f'{garch_result.loglikelihood:.1f}',C['teal']),
    ],className='g-2 mb-3')
    return html.Div([intro, gcard_row,
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_gv,config={'displayModeBar':False}),width=12)],className='mb-2'),
        chain_sec,
        chart_note(
            ('GARCH conditional vol (solid) adapts to volatility clusters. The 60-day amber '
             'forecast mean-reverts toward the long-run unconditional vol level.', 6),
            ('The implied vol smile shows the market prices OTM puts at a premium (skew). '
             'If GARCH vol < smile, a volatility risk premium exists in the market.', 6),
        ),
    ])


# ══ Tab 5: Cross-Company Comparison ══════════════════════════════
def render_comparison(selected_ticker, r, opt_type, T, K_pct):
    intro = slide_intro(5, C['blue'],
        'Cross-Company Comparison — All 10 Names',
        [
            'This tab aggregates key metrics across all ten companies simultaneously. '
            'Prices are normalised to 100 at the start of the five-year window to allow '
            'direct performance comparison regardless of absolute price levels.',
            'Options are priced at-the-money (strike = current spot) using Black-Scholes '
            'with each company\'s own 21-day historical volatility. The GARCH(1,1) model '
            'is fitted independently for each name to compare volatility regimes.',
        ],
        'High-vol tech names (TSLA, NVDA, META) command significantly higher ATM option '
        'premiums than low-vol financials (JPM) or large-caps (MSFT, AAPL). '
        'GARCH persistence (α+β) reveals which stocks stay volatile after shocks.'
    )

    # Load all companies (uses mock data so it's fast)
    rows = []
    norm_data = {}
    hv_series = {}
    for tkr, meta in COMPANIES.items():
        df = _mock_data(tkr)
        S  = float(df['Close'].iloc[-1])
        hv = float(df['HV_21'].iloc[-1])
        K_atm = S  # ATM
        p_bs  = bs_price(S, K_atm, T, r, hv, opt_type)
        p_pct = p_bs / S * 100  # as % of spot
        ret   = df['Log_Return'].dropna()
        try:
            _, gp = fit_garch(ret)
            pers  = gp['alpha'] + gp['beta']
        except Exception:
            pers = float('nan')
        rows.append(dict(Ticker=tkr, Name=meta['name'][:22], Spot=f'${S:.2f}',
                         HV_21=f'{hv*100:.1f}%',
                         ATM_Price=f'${p_bs:.3f}',
                         ATM_Pct=f'{p_pct:.2f}%',
                         Persistence=f'{pers:.4f}',
                         Sector=meta['sector']))
        # Normalised price
        norm_data[tkr] = (df['Close'] / float(df['Close'].iloc[0]) * 100).values
        hv_series[tkr] = df['HV_21'].values
        idx_len = len(df)

    # Normalised price chart
    fig_norm = go.Figure()
    palette = [C['teal'],C['blue'],C['purple'],C['amber'],C['orange'],
               C['coral'],C['green'],C['pink'],C['coral'],C['blue']]
    for i,(tkr,vals) in enumerate(norm_data.items()):
        n = min(len(vals), idx_len)
        x_idx = list(range(n))
        lw = 2.5 if tkr == selected_ticker else 1.0
        op = 1.0 if tkr == selected_ticker else 0.45
        fig_norm.add_trace(go.Scatter(x=x_idx, y=vals[:n], mode='lines', name=tkr,
            line=dict(color=TICKER_COLORS.get(tkr,palette[i%len(palette)]),
                      width=lw), opacity=op))
    apply_layout(fig_norm, 'Normalised Total Return (Base = 100)',
        height=300, legend_h=True)
    fig_norm.update_xaxes(title_text='Trading Days')
    fig_norm.update_yaxes(title_text='Indexed Price')

    # Vol bar chart
    tickers_sorted = sorted(COMPANIES.keys(), key=lambda t: float(rows[[r['Ticker'] for r in rows].index(t)]['HV_21'].strip('%')))
    hv_vals = [float(rows[[r['Ticker'] for r in rows].index(t)]['HV_21'].strip('%')) for t in tickers_sorted]
    colors_bar = [TICKER_COLORS.get(t, C['teal']) for t in tickers_sorted]
    opacity_bar = [1.0 if t == selected_ticker else 0.5 for t in tickers_sorted]
    fig_vol = go.Figure(go.Bar(x=tickers_sorted, y=hv_vals,
        marker=dict(color=colors_bar, opacity=opacity_bar, line=dict(width=0)),
        text=[f'{v:.1f}%' for v in hv_vals], textposition='outside',
        textfont=dict(color=C['muted'], size=11)))
    apply_layout(fig_vol, '21-Day Historical Volatility (Annualised)', height=280)
    fig_vol.update_yaxes(title_text='Vol (%)', ticksuffix='%')

    # ATM option price as % of spot
    atm_pct_vals = [float(rows[[r['Ticker'] for r in rows].index(t)]['ATM_Pct'].strip('%')) for t in tickers_sorted]
    fig_atm = go.Figure(go.Bar(x=tickers_sorted, y=atm_pct_vals,
        marker=dict(color=colors_bar, opacity=opacity_bar, line=dict(width=0)),
        text=[f'{v:.2f}%' for v in atm_pct_vals], textposition='outside',
        textfont=dict(color=C['muted'], size=11)))
    apply_layout(fig_atm, f'ATM {opt_type.title()} Price as % of Spot  (T={T*12:.0f}mo, r={r*100:.2f}%)', height=280)
    fig_atm.update_yaxes(title_text='Premium (%)', ticksuffix='%')

    # Summary table
    tbl = dash_table.DataTable(
        data=rows,
        columns=[{'name': c, 'id': c} for c in ['Ticker', 'Name', 'Sector', 'Spot', 'HV_21', 'ATM_Price', 'ATM_Pct', 'Persistence']],
        style_table={'overflowX': 'auto', 'borderRadius': '8px', 'overflow': 'hidden'},
        style_cell={
            'backgroundColor': C['surface'],
            'color': C['text'],
            'border': f"1px solid {C['border']}",
            'fontFamily': "'JetBrains Mono',monospace",
            'fontSize': '11px',
            'padding': '8px 14px',
            'textAlign': 'left',
        },
        style_header={
            'backgroundColor': C['surface2'],
            'color': C['blue'],
            'fontWeight': '600',
            'fontSize': '10px',
            'letterSpacing': '0.07em',
            'border': f"1px solid {C['border']}",
        },
        style_data_conditional=[  # pyright: ignore[reportArgumentType]
            {
                'if': {'filter_query': '{{Ticker}} = "{}"'.format(selected_ticker)},
                'backgroundColor': f'{C["blue"]}15',
                'border': f'1px solid {C["blue"]}',
            },
        ],
        sort_action='native',
        page_size=10,
    )

    return html.Div([intro,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_norm, config={'displayModeBar':False}), width=12)
        ], className='mb-2'),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_vol, config={'displayModeBar':False}), width=6),
            dbc.Col(dcc.Graph(figure=fig_atm, config={'displayModeBar':False}), width=6),
        ], className='mb-3'),
        html.Div([html.Div('Full Summary Table',style=LBL|{'marginBottom':'10px'}), tbl], style=CARD),
        chart_note(
            ('Normalised returns index all prices to 100 at start. The selected company (brighter line) '
             'is highlighted. High-vol names diverge more from the baseline over time.', 4),
            ('Vol bar: sorted low-to-high. High-vol names (TSLA, NVDA) require larger option premiums. '
             'The selected ticker is shown at full opacity.', 4),
            ('ATM option premium as % of spot directly compares option expensiveness across price levels. '
             'It scales with vol — a cheap stock with high vol can be more expensive to option than a '
             'pricier low-vol name.', 4),
        ),
    ])

print('Tab renderers ready')


# In[7]:


FONTS = ('https://fonts.googleapis.com/css2?'
         'family=Inter:wght@300;400;500;600&'
         'family=JetBrains+Mono:wght@400;500&display=swap')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, FONTS],
    title='Multi-Company Options Dashboard', suppress_callback_exceptions=True)
server = app.server

TAB_BASE = dict(background='transparent', border='none',
    borderBottom='2px solid transparent', padding='11px 22px',
    fontSize='13px', fontWeight='500')
def tab_sel(acc):  return dict(**TAB_BASE, color=acc, borderBottom=f'2px solid {acc}', fontWeight='600')
def tab_idle(acc): return dict(**TAB_BASE, color=C['muted'])

TICKER_OPTIONS = [{'label': f'{t}  —  {m["name"]}', 'value': t} for t, m in COMPANIES.items()]

app.layout = html.Div([

    # ── HEADER ──────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div([
                html.Span('Multi-Company ', style={'color':C['blue'],'fontSize':'24px','fontWeight':'700','fontFamily':MONO}),
                html.Span('Options Pricing Dashboard',
                    style={'color':C['text'],'fontSize':'20px','fontWeight':'500'}),
            ]),
            html.Div('Binomial Model  ·  Black-Scholes  ·  GARCH(1,1)  ·  10-Company Cross-Comparison',
                style={'color':C['muted'],'fontSize':'12px','marginTop':'3px'}),
        ]),
        html.Div([
            html.Div('Muhideen Ogunlowo',
                style={'color':C['blue'],'fontSize':'13px','fontWeight':'600',
                    'letterSpacing':'0.03em','textAlign':'right','marginBottom':'3px'}),
            html.Div('Equities Options Pricing Project',
                style={'color':C['muted'],'fontSize':'11px','textAlign':'right'}),
        ]),
    ], style={'display':'flex','justifyContent':'space-between','alignItems':'center',
        'padding':'16px 36px','background':C['surface'],
        'borderBottom':f"1px solid {C['border']}"}),

    # ── COMPANY SELECTOR + GLOBAL CONTROLS ──────────────────────────────
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div('Company', style=LBL),
                dcc.Dropdown(id='ticker', options=TICKER_OPTIONS, value='DIS',
                    clearable=False,
                    style={'background':C['bg'],'color':C['text'],'border':f"1px solid {C['border']}",
                        'borderRadius':'8px','fontSize':'13px'},
                    className='dark-dropdown'),
            ], width=3),
            dbc.Col([
                html.Div('Risk-Free Rate (%)', style=LBL),
                dcc.Input(id='rf', type='number', value=5.25, min=0, max=20, step=0.05,
                    debounce=True,
                    style={'background':C['bg'],'border':f"1px solid {C['border']}",
                        'color':C['text'],'borderRadius':'8px','padding':'7px 12px',
                        'width':'100%','fontFamily':MONO,'fontSize':'13px'}),
            ], width=2),
            dbc.Col([
                html.Div('Strike — % of Spot', style=LBL),
                dcc.Slider(id='strike-pct', min=70, max=130, step=5, value=100,
                    marks={k:{'label':f'{k}%','style':{'color':C['muted'],'fontSize':'10px'}} for k in [70,80,90,100,110,120,130]},
                    tooltip={'placement':'bottom','always_visible':True}),
            ], width=3),
            dbc.Col([
                html.Div('Maturity (months)', style=LBL),
                dcc.Slider(id='maturity', min=1, max=24, step=1, value=6,
                    marks={k:{'label':str(k),'style':{'color':C['muted'],'fontSize':'10px'}} for k in [1,3,6,9,12,18,24]},
                    tooltip={'placement':'bottom','always_visible':True}),
            ], width=2),
            dbc.Col([
                html.Div('Type / Style', style=LBL),
                dcc.RadioItems(id='opt-type',
                    options=[{'label':' Call','value':'call'},{'label':' Put','value':'put'}],
                    value='call', inline=True,
                    labelStyle={'marginRight':'14px','color':C['text'],'cursor':'pointer','fontSize':'12px'}),
                dcc.RadioItems(id='exercise',
                    options=[{'label':' Euro','value':'european'},{'label':' Am.','value':'american'}],
                    value='european', inline=True,
                    labelStyle={'marginRight':'14px','color':C['text'],'cursor':'pointer','fontSize':'12px'}),
            ], width=2),
        ], className='g-2 align-items-center'),
    ], style={'padding':'14px 36px','background':C['surface'],
        'borderBottom':f"1px solid {C['border']}"}),

    # ── TABS ────────────────────────────────────────────────────────────
    html.Div([
        dcc.Tabs(
            id='tabs',
            value='data',
            style={'borderBottom': f"1px solid {C['border']}"},
            children=[
                dcc.Tab(
                    label='  Market Data',
                    value='data',
                    style=tab_idle(C['teal']),
                    selected_style={**TAB_BASE, 'color': C['teal'], 'borderBottom': f"2px solid {C['teal']}", 'fontWeight': '600'}
                ),
                dcc.Tab(
                    label='  Binomial',
                    value='binom',
                    style=tab_idle(C['purple']),
                    selected_style={**TAB_BASE, 'color': C['purple'], 'borderBottom': f"2px solid {C['purple']}", 'fontWeight': '600'}
                ),
                dcc.Tab(
                    label='  Black-Scholes',
                    value='bs',
                    style=tab_idle(C['amber']),
                    selected_style={**TAB_BASE, 'color': C['amber'], 'borderBottom': f"2px solid {C['amber']}", 'fontWeight': '600'}
                ),
                dcc.Tab(
                    label='  GARCH Forecast',
                    value='garch',
                    style=tab_idle(C['coral']),
                    selected_style={**TAB_BASE, 'color': C['coral'], 'borderBottom': f"2px solid {C['coral']}", 'fontWeight': '600'}
                ),
                dcc.Tab(
                    label='  Cross-Company',
                    value='compare',
                    style=tab_idle(C['blue']),
                    selected_style={**TAB_BASE, 'color': C['blue'], 'borderBottom': f"2px solid {C['blue']}", 'fontWeight': '600'}
                ),
            ],
        ),
        dcc.Loading(
            html.Div(id='content', style={'padding': '22px 36px'}),
            type='circle',
            color=C['blue']
        ),
    ], style={'background': C['bg'], 'minHeight': 'calc(100vh - 160px)'}),

    # ── FOOTER ──────────────────────────────────────────────────────────
    html.Div([
        html.Div('Muhideen Ogunlowo  |  Multi-Company Equities Options Pricing  |  Binomial · Black-Scholes · GARCH(1,1)',
            style={'color':C['muted'],'fontSize':'11px','textAlign':'center'}),
    ], style={'padding':'14px 36px','borderTop':f"1px solid {C['border']}",
        'background':C['surface']}),

    dcc.Store(id='store'),

], style={'background':C['bg'],'fontFamily':'Inter,sans-serif','color':C['text'],'minHeight':'100vh'})

print('App layout ready')


# In[8]:


@app.callback(Output('store','data'),
    Input('ticker','value'), Input('rf','value'))
def load_data(ticker, _):
    df = fetch_data(ticker or 'DIS', '5y')
    df.index = df.index.strftime('%Y-%m-%d')
    return df[['Close','Log_Return','HV_21','HV_63','HV_126']].to_json()

@app.callback(
    Output('content','children'),
    Input('tabs','value'),    Input('store','data'),
    Input('ticker','value'),  Input('strike-pct','value'),
    Input('maturity','value'),Input('opt-type','value'),
    Input('exercise','value'),Input('rf','value'),
)
def render_tab(tab, json_data, ticker, k_pct, months, opt_type, exercise, rf):
    ticker  = ticker or 'DIS'
    acc     = TICKER_COLORS.get(ticker, C['teal'])
    r       = (rf or 5.25) / 100
    T       = (months or 6) / 12
    k_pct   = k_pct or 100

    if tab == 'compare':
        return render_comparison(ticker, r, opt_type, T, k_pct)

    if not json_data:
        return html.Div('Loading...', style={'color':C['muted'],'padding':'40px'})

    df    = pd.read_json(json_data)
    df.index = pd.to_datetime(df.index)
    S     = float(df['Close'].iloc[-1])
    sigma = float(df['HV_21'].iloc[-1])
    K     = S * (k_pct / 100)  # strike as % of spot

    if tab == 'data':  return render_market(df, S, sigma, ticker, acc)
    if tab == 'binom': return render_binomial(S, K, T, r, sigma, opt_type, exercise, ticker, acc)
    if tab == 'bs':    return render_bs(S, K, T, r, sigma, opt_type, exercise, ticker, acc)
    if tab == 'garch': return render_garch(df, S, K, T, r, opt_type, ticker, acc)
    return html.Div()

print('Callbacks registered')


# In[9]:


print('Starting → http://127.0.0.1:8049')
app.run(debug=False, port=8049, jupyter_mode='tab')

