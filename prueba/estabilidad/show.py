import json, os, sys
for n in sys.argv[1:]:
    print("==", n)
    for d in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", n + ".json"))):
        r = d["rho_max_r0,r1,min"]
        print(f"dx={d['dx']:<5} N={d['Nx']}x{d['Ny']}x{d['Nz']} kd={d['kd']} By={d['By']:<7.4g} k1={d['k1']:<6} k3={d['k3']:<6} k4={d['k4']:<6} k6={d['k6']:<8} "
              f"| {d['estado']:<10} it={d['iter']:<6} |g|2={d['grad2']} E={d['E']:<12.6g} E/|Eref|={d['E/|Eref|']:<10.4g} rho_max={r[0] if r else None}")
