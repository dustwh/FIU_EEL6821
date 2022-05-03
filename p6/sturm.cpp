#include <iostream>
#include <cmath>
#include <vector>
#include "sturm.h"
constexpr double rel_error {1.0e-14};   
constexpr int max_pow = 32;
constexpr int max_it = 800;
constexpr double small_enough {1.0e-12}; 
void Sturm::set_parameters()

{
    std::cout << "Please enter order of polynomial: ";
    std::cin >> order;
    std::cout << '\n';
    for (auto i = order; i >= 0; --i) {
        std::cout << "Please enter coefficient number " << i << ": ";
        std::cin >> sturm_seq[0].coef[i];
    }
    std::cout << '\n';
}

std::vector<std::vector<double>> Sturm::get_sturm_sequence()

{
    std::vector<std::vector<double>> seq;
    std::vector<double> coefs;
    
    num_poly = build_sturm();
    for (auto i = order; i >= 0; --i)
	coefs.push_back(sturm_seq[0].coef[i]);
    seq.push_back(coefs);
    for (auto i = 0; i <= num_poly; ++i) {
	coefs.clear();
        for (auto j = sturm_seq[i].ord; j >= 0; --j)
	    coefs.push_back(sturm_seq[i].coef[j]);
	seq.push_back(coefs);
    }
    return seq;
}

void Sturm::show_sturm_sequence(const std::vector<std::vector<double>> &seq)
{
    std::cout << "Sturm sequence for:\n";
    std::cout << std::fixed;
    auto first = true;
    for (const auto &poly : seq) {
	for(const auto &coef : poly)
	    std::cout << coef << ' ';
	if (first) { std::cout << "\n"; first = false;}
        std::cout << "\n";
    }
}

int Sturm::build_sturm()

{
    double f, *fp, *fc;
    Poly *sp;

    sturm_seq[0].ord = order;
    sturm_seq[1].ord = order - 1;

    f = fabs(sturm_seq[0].coef[order] * order);
    fp = sturm_seq[1].coef;
    fc = sturm_seq[0].coef + 1;
    for (auto i = 1; i <= order; i++)
        *fp++ = *fc++ *i / f;

    for (sp = sturm_seq + 2; modp(sp - 2, sp - 1, sp); sp++) {
        f = -fabs(sp->coef[sp->ord]);
        for (fp = &sp->coef[sp->ord]; fp >= sp->coef; fp--)
            *fp /= f;
    }

    sp->coef[0] = -sp->coef[0]; 

    return sp - sturm_seq;
}

int Sturm::modp(Poly *u, Poly *v, Poly *r)

{
    int k, j;
    double *nr, *end, *uc;

    nr = r->coef;
    end = &u->coef[u->ord];

    uc = u->coef;
    while (uc <= end)
        *nr++ = *uc++;

    if (v->coef[v->ord] < 0.0) {
        for (k = u->ord - v->ord - 1; k >= 0; k -= 2)
            r->coef[k] = -r->coef[k];
        for (k = u->ord - v->ord; k >= 0; k--)
            for (j = v->ord + k - 1; j >= k; j--)
                r->coef[j] = -r->coef[j] - r->coef[v->ord + k] * v->coef[j - k];
    }
    else {
        for (k = u->ord - v->ord; k >= 0; k--)
            for (j = v->ord + k - 1; j >= k; j--)
                r->coef[j] -= r->coef[v->ord + k] * v->coef[j - k];
    }

    k = v->ord - 1;
    while (k >= 0 && fabs(r->coef[k]) < small_enough) {
        r->coef[k] = 0.0;
        k--;
    }

    r->ord = (k < 0) ? 0 : k;

    return r->ord;
}

std::vector<double> Sturm::get_real_roots()
{
    nroots = num_roots();
    if (nroots == 0) {
	std::cout << "solve: no real roots\n";
        exit (0);
    }
    
    nchanges = num_changes(min);
    for (auto i = 0; nchanges != atmin && i != max_pow; ++i) {
        min *= 10.0;
        nchanges = num_changes(min);
    }
    if (nchanges != atmin) {
	std::cout << "solve: unable to bracket all negative roots\n";
        atmin = nchanges;
    }

    nchanges = num_changes(max);
    for (auto i = 0; nchanges != atmax && i != max_pow; ++i) {
        max *= 10.0;
        nchanges = num_changes(max);
    }
    if (nchanges != atmax) {
	std::cout << "solve: unable to bracket all positive roots\n";
        atmax = nchanges;
    }
    nroots = atmin - atmax;

    bisect(min, max, atmin, atmax, roots);

    std::vector<double> roots_vec;
    for (auto i = 0; i != nroots; ++i)
        roots_vec.push_back(roots[i]);

    return roots_vec;
}

int Sturm::num_roots()

{
    int atposinf = 0, atneginf = 0;
    Poly *s;
    double f, lf;

    lf = sturm_seq[0].coef[sturm_seq[0].ord];

    for (s = sturm_seq + 1; s <= sturm_seq + num_poly; ++s) {
        f = s->coef[s->ord];
        if (lf == 0.0 || lf * f < 0) ++atposinf;
        lf = f;
    }

    if (sturm_seq[0].ord & 1) lf = -sturm_seq[0].coef[sturm_seq[0].ord];
    else lf = sturm_seq[0].coef[sturm_seq[0].ord];

    for (s = sturm_seq + 1; s <= sturm_seq + num_poly; ++s) {
        if (s->ord & 1) f = -s->coef[s->ord];
        else f = s->coef[s->ord];
        if (lf == 0.0 || lf * f < 0) ++atneginf;
        lf = f;
    }

    atmin = atneginf;
    atmax = atposinf;

    return atneginf - atposinf;
}

int Sturm::num_changes(const double& a)

{
    double f, lf;
    Poly *s;
    int changes = 0;

    lf = eval_poly(sturm_seq[0].ord, sturm_seq[0].coef, a);

    for (s = sturm_seq + 1; s <= sturm_seq + num_poly; s++) {
        f = eval_poly(s->ord, s->coef, a);
        if (lf == 0.0 || lf * f < 0) ++changes;
        lf = f;
    }

    return changes;
}

double Sturm::eval_poly(int ord, double *coef, double x)
{

    double *fp, f;
    fp = &coef[ord];
    f = *fp;
    for (fp--; fp >= coef; fp--)
        f = x * f + *fp;
    return f;
}

void Sturm::bisect(double min, double  max, const int &atmin, const int &atmax, double *roots)

{
    double mid;
    int n1 = 0, n2 = 0, its, atmid, nroot;

    if ((nroot = atmin - atmax) == 1) {

        if (mod_rf(sturm_seq->ord, sturm_seq->coef, min, max, &roots[0]))
            return;

        for (its = 0; its < max_it; its++) {
            mid = (min + max) / 2;
            atmid = num_changes(mid);
            if (fabs(mid) > rel_error) {
                if (fabs((max - min) / mid) < rel_error) {
                    roots[0] = mid;
                    return;
                }
            }
            else if (fabs(max - min) < rel_error) {
                roots[0] = mid;
                return;
            }
            if ((atmin - atmid) == 0) min = mid;
            else max = mid;
        }
	if (its == max_it) {
            std::cerr << "bisect: overflow min " << min << " max " << max
                      << " diff " << max - min << " nroot " << nroot << " n1 "
                      << n1 << " n2 " << n2 << "\n";
            roots[0] = mid;
        }
        return;
    }

    for (its = 0; its < max_it; its++) {
        mid = (min + max) / 2;
        atmid = num_changes(mid);
        n1 = atmin - atmid;
        n2 = atmid - atmax;
        if (n1 != 0 && n2 != 0) {
            bisect(min, mid, atmin, atmid, roots);
            bisect(mid, max, atmid, atmax, &roots[n1]);
            break;
        }

        if (n1 == 0) min = mid;
        else max = mid;
    }

    if (its == max_it) {
	std::cerr << "bisect: roots too close together\n";
        std::cerr << "bisect: overflow min " << min << " max " << max
                  << " diff " << max - min << " nroot " << nroot << " n1 " << n1
                  << " n2 " << n2 << "\n";
        for (n1 = atmax; n1 < atmin; n1++) roots[n1 - atmax] = mid;
    }
}

int Sturm::mod_rf(int ord, double *coef, double a, double b, double *val)
{

    int its;
    double fa, fb, x, fx, lfx;
    double *fp, *scoef, *ecoef;

    scoef = coef;
    ecoef = &coef[ord];

    fb = fa = *ecoef;
    for (fp = ecoef - 1; fp >= scoef; fp--) {
        fa = a * fa + *fp;
        fb = b * fb + *fp;
    }

    if (fa * fb > 0.0)
        return 0;
    if (fabs(fa) < rel_error) {
        *val = a;
        return 1;
    }
    if (fabs(fb) < rel_error) {
        *val = b;
        return 1;
    }

    lfx = fa;

    for (its = 0; its < max_it; its++) {
        x = (fb * a - fa * b) / (fb - fa);
        fx = *ecoef;
        for (fp = ecoef - 1; fp >= scoef; fp--)
            fx = x * fx + *fp;
        if (fabs(x) > rel_error) {
            if (fabs(fx / x) < rel_error) {
                *val = x;
                return 1;
            }
        }
        else if (fabs(fx) < rel_error) {
            *val = x;
            return 1;
        }
        if ((fa * fx) < 0) {
            b = x;
            fb = fx;
            if ((lfx * fx) > 0)
                fa /= 2;
        }
        else {
            a = x;
            fa = fx;
            if ((lfx * fx) > 0)
                fb /= 2;
        }

        lfx = fx;
    }

    std::cerr << "mod_rf overflow " << a << " " << b << " " << fx << "\n";

    return 0;
}

void Sturm::show_roots(const std::vector<double> &roots)

{
    if (roots.size() == 1) {
        std::cout << "\n1 distinct real root at x = " << roots.front() << "\n";
    }
    else {
        std::cout << "\n" << roots.size() << " distinct real roots for x: ";
        for (const auto& root : roots) std::cout << root << ' ';
        std::cout << '\n';
    }
}