"""Regenerate tests/fixtures/s5-r-golden.json: deterministic data sets and R 4.3's answers (lm, glm, arima, acf, pacf).
Run on the host (needs Rscript, stdlib Python only):  python3 scripts/s5-r-golden.py
The fixture holds the data too, so tests never need R; the recipes use the same data against rcli/lmstats."""
import json, math, pathlib, subprocess, sys, tempfile
root = pathlib.Path(__file__).resolve().parents[1]
OUT = root / 'tests/fixtures/s5-r-golden.json'


def lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield (state >> 11) / float(1 << 53)


def normal(seed):
    u = lcg(seed)
    while True:
        a, b = next(u), next(u)
        r = math.sqrt(-2.0 * math.log(max(a, 1e-300)))
        yield r * math.cos(2 * math.pi * b)
        yield r * math.sin(2 * math.pi * b)


def build_data():
    n = 40
    noise, uni = normal(20261015), lcg(77)
    x1 = [round((i * 37 % 41) / 4.1, 6) for i in range(n)]
    x2 = [round((i * 53 % 47) / 9.4 - 2.0, 6) for i in range(n)]
    g = ['abc'[(i * 5 + i // 3) % 3] for i in range(n)]
    w = [1.0 + (i % 4) / 2.0 for i in range(n)]
    effect = {'a': 0.7, 'b': 0.0, 'c': -0.4}
    y = [round(1 + 2 * x1[i] - 0.5 * x2[i] + effect[g[i]] + 0.1 * x1[i] * x2[i] + 0.6 * next(noise), 6) for i in range(n)]
    lam = [math.exp(0.5 + 0.08 * x1[i] - 0.1 * x2[i] + 0.3 * effect[g[i]]) for i in range(n)]
    count = [max(0, int(round(lam[i] + 1.2 * next(noise)))) for i in range(n)]
    binary = [1 if 1 / (1 + math.exp(-(-0.3 + 0.35 * x1[i] - 0.4 * x2[i]))) > next(uni) else 0 for i in range(n)]
    gam = [round(math.exp(0.2 + 0.1 * x1[i]) * (0.6 + 0.8 * next(uni)), 6) for i in range(n)]
    gn = normal(424242)
    series = []
    previous = 0.0
    for _ in range(150):
        previous = 0.6 * previous + next(gn)
        series.append(round(previous, 6))
    series = series[30:]
    return {'x1': x1, 'x2': x2, 'g': g, 'w': w, 'y': y, 'count': count, 'binary': binary, 'gam': gam, 'series': series}


R_TEMPLATE = r'''
num <- function(v) paste(sprintf("%.17g", as.numeric(v)), collapse=" ")
emit <- function(key, v) cat(key, "\t", num(v), "\n", sep="")
d <- data.frame(x1=c(@x1@), x2=c(@x2@), g=factor(c(@g@)), w=c(@w@), y=c(@y@), count=c(@count@), binary=c(@binary@), gam=c(@gam@))
d$g <- relevel(d$g, ref="b")
ci <- function(fit, level=0.95) confint.default(fit, level=level)
f <- lm(y ~ x1 + x2 + g + x1:x2, data=d); s <- summary(f)
emit("lm.names_count", length(coef(f))); emit("lm.coef", coef(f)); emit("lm.se", s$coefficients[,2]); emit("lm.t", s$coefficients[,3]); emit("lm.p", s$coefficients[,4])
emit("lm.ci", t(confint(f))); emit("lm.r2", s$r.squared); emit("lm.adjr2", s$adj.r.squared); emit("lm.sigma", s$sigma); emit("lm.fstat", s$fstatistic[1])
emit("lm.loglik", as.numeric(logLik(f))); emit("lm.aic", AIC(f)); emit("lm.bic", BIC(f)); emit("lm.df", f$df.residual)
fw <- lm(y ~ x1 + x2 + g, data=d, weights=w); sw <- summary(fw)
emit("wls.coef", coef(fw)); emit("wls.se", sw$coefficients[,2]); emit("wls.r2", sw$r.squared); emit("wls.sigma", sw$sigma)
fn <- lm(y ~ 0 + x1 + x2, data=d); sn <- summary(fn)
emit("nocon.coef", coef(fn)); emit("nocon.se", sn$coefficients[,2]); emit("nocon.r2", sn$r.squared)
gp <- glm(count ~ x1 + x2 + g, family=poisson(), data=d); sp <- summary(gp)
emit("poisson.coef", coef(gp)); emit("poisson.se", sp$coefficients[,2]); emit("poisson.z", sp$coefficients[,3]); emit("poisson.deviance", gp$deviance); emit("poisson.null", gp$null.deviance)
emit("poisson.aic", AIC(gp)); emit("poisson.bic", BIC(gp)); emit("poisson.loglik", as.numeric(logLik(gp))); emit("poisson.iter", gp$iter)
gb <- glm(binary ~ x1 + x2, family=binomial(), data=d); sb <- summary(gb)
emit("binomial.coef", coef(gb)); emit("binomial.se", sb$coefficients[,2]); emit("binomial.deviance", gb$deviance); emit("binomial.null", gb$null.deviance)
emit("binomial.aic", AIC(gb)); emit("binomial.bic", BIC(gb)); emit("binomial.loglik", as.numeric(logLik(gb)))
gg <- glm(gam ~ x1, family=Gamma(link="log"), data=d); sg <- summary(gg)
emit("gamma.coef", coef(gg)); emit("gamma.se", sg$coefficients[,2]); emit("gamma.deviance", gg$deviance); emit("gamma.null", gg$null.deviance); emit("gamma.dispersion", sg$dispersion)
emit("gamma.aic", AIC(gg)); emit("gamma.loglik", as.numeric(logLik(gg)))
d$g <- relevel(d$g, ref="a")
fa <- lm(y ~ x1 + x2 + g + x1:x2, data=d); sa <- summary(fa)
emit("lm_a.coef", coef(fa)); emit("lm_a.se", sa$coefficients[,2]); emit("lm_a.r2", sa$r.squared); emit("lm_a.sigma", sa$sigma)
gp2 <- glm(count ~ x1 + x2, family=poisson(), data=d); sp2 <- summary(gp2)
emit("poisson2.coef", coef(gp2)); emit("poisson2.se", sp2$coefficients[,2]); emit("poisson2.deviance", gp2$deviance); emit("poisson2.aic", AIC(gp2))
x <- c(@series@)
a1 <- arima(x, order=c(1,0,1), include.mean=FALSE, method="ML"); emit("arma11.coef", coef(a1)); emit("arma11.se", sqrt(diag(a1$var.coef))); emit("arma11.sigma2", a1$sigma2); emit("arma11.loglik", a1$loglik); emit("arma11.aic", a1$aic)
p1 <- predict(a1, n.ahead=5); emit("arma11.pred", p1$pred); emit("arma11.pred.se", p1$se)
a2 <- arima(x, order=c(2,0,0), include.mean=TRUE, method="ML"); emit("ar2.coef", coef(a2)); emit("ar2.se", sqrt(diag(a2$var.coef))); emit("ar2.sigma2", a2$sigma2); emit("ar2.loglik", a2$loglik); emit("ar2.aic", a2$aic)
a3 <- arima(x, order=c(1,1,1), method="ML"); emit("arima111.coef", coef(a3)); emit("arima111.sigma2", a3$sigma2); emit("arima111.loglik", a3$loglik); emit("arima111.aic", a3$aic)
emit("acf", acf(x, lag.max=12, plot=FALSE)$acf[,1,1]); emit("pacf", pacf(x, lag.max=12, plot=FALSE)$acf[,1,1])
'''


def literal(values):
    return ','.join(("'%s'" % v) if isinstance(v, str) else repr(float(v)) if isinstance(v, float) else str(v) for v in values)


def main():
    data = build_data()
    code = R_TEMPLATE
    for key, values in data.items():
        code = code.replace('@' + key + '@', literal(values))
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / 'golden.R'
        path.write_text(code)
        out = subprocess.run(['Rscript', '--vanilla', str(path)], capture_output=True, text=True)
    if out.returncode:
        sys.exit(out.stderr)
    golden = {}
    for line in out.stdout.splitlines():
        key, values = line.split('\t')
        golden[key] = [float(v) for v in values.split()]
    OUT.write_text(json.dumps({'generator': 'scripts/s5-r-golden.py, R 4.3.3', 'data': data, 'r': golden}, indent=0) + '\n')
    print(f'wrote {OUT}: {len(golden)} R vectors, {len(data["y"])} regression rows, {len(data["series"])} series points')


main()
