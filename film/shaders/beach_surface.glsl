// ------------------------------------------------------------------ the beach
vec2 polUV(float th,float rho){ return vec2((th-uPol.x)/(uPol.y*float(uNS.x)),(rho-uPol.z)/(uPol.w*float(uNS.y))); }
vec2 terUV(float th,float rho){ return vec2((th-uTer.x)/(uTer.y-uTer.x),(rho-uTer.z)/(uTer.w-uTer.z)); }

// march along the column of this azimuth: rows are ln(range), so it is one lookup a step
bool marchBeach(float th,float tanEl,float fw,out float rhoHit){
  rhoHit=0.;
  float rs=(uH-uTop)/max(-tanEl,1e-5);
  float rho=log(max(rs,exp(uPol.z)));
  float rhoEnd=log(uRmax);
  if(rho>=rhoEnd) return false;
  float cell=uPol.w;
  float ds=max(cell,.35*fw/max(-tanEl,1e-5));
  float lod=0.;
  float bd=cell*uBlock;
  float ux=(th-uPol.x)/(uPol.y*float(uNS.x));
  float vScale=1./(cell*float(uNS.y));
  float prevRho=rho,prevGap=1e3;
  int bx=int(ux*float(uNS.x));
  for(int i=0;i<400;i++){
    if(rho>=rhoEnd) return false;
    float fj=(rho-uPol.z)/bd;
    float blk=floor(fj);
    float rhoB=uPol.z+(blk+1.)*bd;
    float rb=exp(rhoB);
    float yb=uH+rb*tanEl-rb*rb*uCurv;
    float hm=texelFetch(uHmax,ivec2(bx,int(blk)),0).r;
    if(yb>hm){ prevRho=rhoB; prevGap=1e3; rho=rhoB+1e-5; continue; }
    float r=exp(rho);
    float y=uH+r*tanEl-r*r*uCurv;
    float s=textureLod(uR0,vec2(ux,(rho-uPol.z)*vScale),lod).r;
    float gap=y-s;
    if(gap<=0.){
      float a=prevRho,b=rho,ga=prevGap,gb=gap;
      if(ga>100.){ rhoHit=rho; return true; }
      for(int k=0;k<4;k++){
        float m=a+(b-a)*clamp(ga/max(ga-gb,1e-5),.05,.95);
        float rm=exp(m);
        float gm=uH+rm*tanEl-rm*rm*uCurv-textureLod(uR0,vec2(ux,(m-uPol.z)*vScale),lod).r;
        if(gm>0.){ a=m; ga=gm; } else { b=m; gb=gm; }
      }
      rhoHit=a+(b-a)*ga/max(ga-gb,1e-5);
      return true;
    }
    prevRho=rho; prevGap=gap;
    rho+=ds;
  }
  rhoHit=rho;
  return true;
}

// worley distances: F1, F2-F1
vec2 worley(vec2 g,float seed){
  vec2 c=floor(g);
  float f1=9.,f2=9.;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 id=c+vec2(float(i),float(j));
    vec2 r=id+hash22(id+seed)-g;
    float d=dot(r,r);
    if(d<f1){ f2=f1; f1=d; } else if(d<f2) f2=d;
  }
  f1=sqrt(f1); f2=sqrt(f2);
  return vec2(f1,f2-f1);
}

// coverage of foam from the baked field. The channels hold 7, 22, 50 and 82 percent of the area, so
// the number that goes in is the share of the pixel that comes out, at every distance.
float laceLevel(vec4 c,float a){
  if(a<.07) return c.r*a/.07;
  if(a<.22) return mix(c.r,c.g,(a-.07)/.15);
  if(a<.50) return mix(c.g,c.b,(a-.22)/.28);
  if(a<.82) return mix(c.b,c.a,(a-.50)/.32);
  return mix(c.a,1.,(a-.82)/.18);
}
// the height of the field that leaves that same share above it
float laceHeight(float a){
  if(a<.07) return mix(1.05,uLaceTh.x,a/.07);
  if(a<.22) return mix(uLaceTh.x,uLaceTh.y,(a-.07)/.15);
  if(a<.50) return mix(uLaceTh.y,uLaceTh.z,(a-.22)/.28);
  if(a<.82) return mix(uLaceTh.z,uLaceTh.w,(a-.50)/.32);
  return mix(uLaceTh.w,-.08,(a-.82)/.18);
}
const mat2 LR1=mat2(.766,.643,-.643,.766),LR2=mat2(-.309,.951,-.951,-.309),LR3=mat2(.122,-.993,.993,.122);
// A raft of foam is not a sheet: it heaps. With the sun low behind it the side of a heap towards the camera is in
// shade and only its top catches the light, at every scale from a metre or two down to a fist.
vec2 foamHeaps(vec2 q,vec2 ax,vec2 ay){
  vec2 Lh=normalize(uSun.xz);
  const float L=8.;
  float k1=1./(L*1.7),k2=1./(L*.62);
  vec2 gx1=LR2*ax*k1,gy1=LR2*ay*k1,gx2=LR3*ax*k2,gy2=LR3*ay*k2;
  vec2 u1=LR2*q*k1+.13,u2=LR3*q*k2+.41;
  vec2 a0=textureGrad(uLaceN,u1,gx1,gy1).rg,b0=textureGrad(uLaceN,u2,gx2,gy2).rg;
  vec2 a1=textureGrad(uLaceN,u1+LR2*Lh*k1*.42,gx1,gy1).rg,b1=textureGrad(uLaceN,u2+LR3*Lh*k2*.16,gx2,gy2).rg;
  float h0=a0.r*.4+a0.g*.27+b0.r*.2+b0.g*.13;
  float h1=a1.r*.4+a1.g*.27+b1.r*.2+b1.g*.13;
  return vec2(h0,(h0-h1));
}
float foamLace(vec2 q,float F,vec2 ax,vec2 ay,out float amt,out vec4 rel){
  const float L=8.;
  // rafts and leads: soft fields far larger than any strand, so foam has structure at the scale of the surf zone
  vec2 u3=LR3*q/(L*3.);
  float big=textureGrad(uLaceN,u3,LR3*ax/(L*3.),LR3*ay/(L*3.)).r;
  vec2 u4=LR1*q/(L*1.17)+.37;
  float mid=textureGrad(uLaceN,u4,LR1*ax/(L*1.17),LR1*ay/(L*1.17)).g;
  // dense only right behind the bore, then thinning: an old raft is mostly leads
  amt=min(pow(F,1.4)*(.1+.85*big+.42*mid),.94);
  if(amt<=.004){ rel=vec4(0.); return 0.; }
  const float M=.73;
  vec2 u1=LR1*q/uLaceL,u2=LR2*q/(uLaceL*M)+.19;
  vec4 c1=textureGrad(uLace,u1,LR1*ax/uLaceL,LR1*ay/uLaceL);
  vec4 c2=textureGrad(uLace,u2,LR2*ax/(uLaceL*M),LR2*ay/(uLaceL*M));
  rel=textureGrad(uLaceN,u1,LR1*ax/uLaceL,LR1*ay/uLaceL);
  float s=smoothstep(.38,.62,big*.5+mid*.5);
  float l=mix(laceLevel(c1,amt),laceLevel(c2,amt),s);
  // near enough to see single strands: the field itself, cut at exactly this amount, so the amount
  // moves the edge of the foam instead of fading a band of it
  float near_=1.-smoothstep(.008,.03,max(length(ax),length(ay)));
  if(near_>0.){
    float pB=textureGrad(uLaceN,u2,LR2*ax/(uLaceL*M),LR2*ay/(uLaceL*M)).b;
    float th=laceHeight(amt);
    l=mix(l,mix(smoothstep(th-.05,th+.05,rel.b),smoothstep(th-.05,th+.05,pB),s),near_);
  }
  // strands are made of bubbles: up close they are grainy and broken, never a drawn line
  return l*mix(mix(.7,1.1,rel.a),1.,smoothstep(.55,.95,amt));
}

// four lookups for a smooth bicubic: the corners of the cells must not show in what is cut from a field
vec4 bsplineL(sampler2D t,vec2 uv,vec2 n,float l){
  vec2 g=uv*n-.5,i=floor(g),f=g-i,f2=f*f,f3=f2*f;
  vec2 w0=(1.-3.*f+3.*f2-f3)/6.,w1=(4.-6.*f2+3.*f3)/6.,w2=(1.+3.*f+3.*f2-3.*f3)/6.,w3=f3/6.;
  vec2 g0=w0+w1,g1=w2+w3;
  vec2 p0=(i-.5+w1/g0)/n,p1=(i+1.5+w3/g1)/n;
  return g0.y*(g0.x*textureLod(t,vec2(p0.x,p0.y),l)+g1.x*textureLod(t,vec2(p1.x,p0.y),l))
        +g1.y*(g0.x*textureLod(t,vec2(p0.x,p1.y),l)+g1.x*textureLod(t,vec2(p1.x,p1.y),l));
}
vec4 bspline(sampler2D t,vec2 uv,vec2 n){ return bsplineL(t,uv,n,0.); }

float skyT;    // how much of the sun the clouds let through right now

vec4 stackAt(float az,float el,vec2 gx,vec2 gy){
  vec2 w=(vec2(az,el)-uStackWin.xz)/(uStackWin.yw-uStackWin.xz);
  if(w.x<=0.||w.x>=1.||w.y<=0.||w.y>=1.) return vec4(0.);
  return textureGrad(uStacks,w,gx/(uStackWin.yw-uStackWin.xz),gy/(uStackWin.yw-uStackWin.xz));
}

// the sky seen in something smooth: the camera's window at its own resolution, blurred by the
// spread of the facet slopes, much more up-and-down than sideways at these angles
vec3 skyMirror(vec3 r,float az_s,float el_s){
  r.y=max(r.y,.0008);
  r=normalize(r);
  float az=atan(r.x,r.z),el=asin(r.y);
  float inW=step(uSkyWin.x,az)*step(az,uSkyWin.y)*step(el,uSkyWin.w);
  if(inW<.5||max(az_s,el_s)>.09){
    float lod=max(log2(max(sqrt(az_s*el_s),.0005)/.00307),0.);
    return textureLod(uEnv,vec2(az/(2.*PI)+.5,el/(.5*PI)),lod).rgb;
  }
  vec3 acc=vec3(0.); float T=1.;
  vec2 gx=vec2(az_s/(uSkyWin.y-uSkyWin.x),0.),gy=vec2(0.,el_s/(uSkyWin.w-uSkyWin.z));
  for(int k=0;k<3;k++){
    vec3 dk=normalize(r+uDrift[k]*max(r.y,0.));
    vec2 w=vec2((atan(dk.x,dk.z)-uSkyWin.x)/(uSkyWin.y-uSkyWin.x),(asin(clamp(dk.y,0.,1.))-uSkyWin.z)/(uSkyWin.w-uSkyWin.z));
    vec4 c=textureGrad(uCloud,vec3(w,float(k)),gx,gy);
    acc+=T*c.rgb; T*=c.a;
  }
  vec3 cc=acc+T*texture(uBg,panoUV(r)).rgb;
  vec4 st=stackAt(az,el,vec2(az_s,0.),vec2(0.,el_s));
  return cc*(1.-st.a)+st.rgb;
}

// Salt in the air over the surf: the camera is so low that a ray spends most of its length inside
// the few metres above the water. Smooth by construction: this is haze, not a texture.
float sprayAlong(float th,float tanEl,float rhoEnd){
  const int N=9;
  const float SPAN=1.15;
  float ux=(th-uPol.x)/(uPol.y*float(uNS.x));
  float vs=1./(uPol.w*float(uNS.y));
  float dr=SPAN/float(N);
  float tau=0.;
  for(int i=0;i<N;i++){
    float rho=max(rhoEnd-SPAN+(float(i)+.5)*dr,uPol.z);
    float r=exp(rho);
    float y=uH+r*tanEl;
    float src=textureLod(uR1,vec2(ux,(rho-uPol.z)*vs),4.5).b;
    // it comes off the lip in rags, more from one section than the next, and blows away
    vec2 qa=r*vec2(sin(th),cos(th))+vec2(uT*.5,-uT*1.2);
    float rag=textureLod(uLaceN,qa/41.,3.).r*.6+textureLod(uLaceN,qa/13.+.4,3.).r*.4;
    rag=smoothstep(.34,.7,rag);
    tau+=src*(.1+1.9*rag)*exp(-max(y,0.)/(1.5+2.6*rag))*dr*r;
  }
  return tau;
}

vec3 shadeBeach(vec3 d,vec3 dx,vec3 dy,float fw,float th,float rho,float tanEl,out float dist){
  vec3 V=-d;
  float r=exp(rho);
  float ch=sqrt(1.+tanEl*tanEl);
  dist=r*ch;
  vec2 nr=vec2(sin(th),cos(th)),nt=vec2(nr.y,-nr.x);
  vec2 q=r*nr;
  vec2 uv0=polUV(th,rho);
  float gr=max(-tanEl,.0012);
  float wa=r*fw,wb=wa/gr;                    // pixel footprint across and along the ray, metres
  float cell=uPol.w;
  vec2 gx=vec2(fw/(uPol.y*float(uNS.x)),0.),gy=vec2(0.,fw/gr/(cell*float(uNS.y)));
  // the slope of the resolved surface: never through the footprint on the plane, because a wave
  // face turned to the camera is many pixels tall even where the flat water around it is not one
  vec2 d0=vec2(.75/float(uNS.x),0.),d1=vec2(0.,.75/float(uNS.y));
  float e_e=textureLod(uR0,uv0+d0,0.).r,e_w=textureLod(uR0,uv0-d0,0.).r;
  float e_n=textureLod(uR0,uv0+d1,0.).r,e_s=textureLod(uR0,uv0-d1,0.).r;
  float dth=2.*d0.x*uPol.y*float(uNS.x),drho=2.*d1.y*cell*float(uNS.y);
  vec2 sim=((e_e-e_w)/dth*nt+(e_n-e_s)/drho*nr)/r;
  // but over no less than what a pixel covers of the water, either: taken over less it is a lottery from one
  // row of pixels to the next, and the rows come out combed
  if(uHero>0.){
    vec2 dc=vec2(0.,6./float(uNS.y));
    float sc=(textureLod(uR0,uv0+dc,2.5).r-textureLod(uR0,uv0-dc,2.5).r)/(12.*cell*r);
    float kw=clamp(.5*wa/(gr+max(sc,0.))/(r*cell),.75,14.);
    if(kw>.8){
      float lk=log2(kw/.75);
      vec2 dr=vec2(0.,kw/float(uNS.y)),dt=vec2(1.5/float(uNS.x),0.);
      float ls=min(lk,2.6);
      float sR=(textureLod(uR0,uv0+dr,ls).r-textureLod(uR0,uv0-dr,ls).r)/(2.*kw*cell*r);
      float sT=(textureLod(uR0,uv0+dt,ls).r-textureLod(uR0,uv0-dt,ls).r)/(3.*uPol.y*r);
      sim=mix(sim,sT*nt+sR*nr,uHero);
    }
  }
  // close to, a slope taken across the cells shows the cells, and the solver's fronts are creases where
  // water would be round: the same slope, taken smoothly at about the scale the water holds a shape
  vec4 Rs=vec4(0.);
  if(r<45.){
    const float LS=2.;
    vec2 nl=vec2(uNS)/4.;
    vec2 d4=vec2(4./float(uNS.x),0.),d5=vec2(0.,4./float(uNS.y));
    float a0=bsplineL(uR0,uv0+d4,nl,LS).r,a1=bsplineL(uR0,uv0-d4,nl,LS).r;
    float a2=bsplineL(uR0,uv0+d5,nl,LS).r,a3=bsplineL(uR0,uv0-d5,nl,LS).r;
    float k4=8.*uPol.y;
    vec2 simS=((a0-a1)/k4*nt+(a2-a3)/k4*nr)/r;
    sim=mix(simS*mix(.55,1.,smoothstep(11.,22.,r)),sim,smoothstep(18.,45.,r));
    Rs=bspline(uR0,uv0,vec2(uNS));
  }
  float grEff=gr+max(dot(sim,nr),0.);
  wb=wa/grEff;
  gy=vec2(0.,fw/grEff/(cell*float(uNS.y)));
  // the edge of the water between cells: wherever it runs, the sand decides, not the grid
  vec2 uvW=uv0;
  if(r<70.){
    vec2 nq=q/.21;
    vec2 wv=vec2(vnoise(nq)+.5*vnoise(nq*2.3+5.),vnoise(nq+9.)+.5*vnoise(nq*2.3+13.))/1.5-.5;
    uvW+=wv*2.6*(1.-smoothstep(40.,70.,r))/vec2(uNS);
  }
  vec4 R=textureGrad(uR0,uvW,gx,gy);
  float eta=R.r,h=R.g;
  vec2 vel=R.ba;
  vec4 B=textureGrad(uR1,uvW,gx,gy);
  float Fo=B.r;
  // on a face the lip's foam is smeared sideways a little: the columns of the grid must not print through
  if(B.a>.01) Fo=mix(Fo,textureGrad(uR1,uvW,gx*5.,gy).r,smoothstep(.01,.08,B.a));
  if(r<45.) Fo=mix(bspline(uR1,uvW,vec2(uNS)).r,Fo,smoothstep(25.,45.,r));
  // A bore's foam is drawn out straight along its run and the grid's columns print into that: ribs a few cells apart, at
  // one even spacing, the comb on the faces and round the hook. Taken wider across the run they are gone; what the raft
  // keeps of structure across the run is the lace's. Wandering the lookup breaks up any that would still line up.
  if(uHero>0.&&r>18.){
    vec2 nw=vec2(vnoise(q*.37)-.5,vnoise(q*.37+vec2(4.1,9.3))-.5);
    vec2 uvF=uvW+vec2(nw.x*18.,nw.y*5.)/vec2(uNS)*mix(.4,1.,smoothstep(18.,40.,r));
    float nb=mix(4.,9.,smoothstep(30.,110.,r));
    float wide=textureGrad(uR1,uvF,max(gx,vec2(nb/float(uNS.x),0.)),gy).r;
    Fo=mix(Fo,wide,uHero*smoothstep(18.,34.,r));
  }
  // sand drains unevenly around the line the water last reached: no ruled edge between wet and damp
  float Wet=mix(textureLod(uR1,uvW,2.6).g,B.g,smoothstep(30.,60.,r));
  vec4 dm=r<70.?bspline(uMat,uv0,vec2(uNS)):texture(uMat,uv0);
  // the coordinates update at 60 Hz, the water at 360: carry them the rest of the way, or at 24 frames a
  // second the foam walks in steps of two and three
  dm-=vec4(vel,vel)*uMatLag;
  // the water's bulk moves the lace; its small eddies would only draw a net out into brush strokes
  if(uHero>0.&&r<70.){
    vec2 k=1.1/(r*cell)/vec2(uNS);
    vec4 ml=.25*(texture(uMat,uv0+vec2(k.x,0.))+texture(uMat,uv0-vec2(k.x,0.))+texture(uMat,uv0+vec2(0.,k.y))+texture(uMat,uv0-vec2(0.,k.y)))-vec4(vel,vel)*uMatLag;
    dm=mix(dm,ml,uHero*.9);
  }

  // the sand under it all
  vec2 tu=terUV(th,rho);
  vec2 tgx=vec2(fw/(uTer.y-uTer.x),0.),tgy=vec2(0.,fw/gr/(uTer.w-uTer.z));
  vec4 T0=vec4(0.,0.,0.,1.),T1=vec4(0.);
  if(tu.y<1.){
    T0=textureGrad(uT0,tu,tgx,tgy);
    T1=textureGrad(uT1,tu,tgx,tgy);
  }
  float far=smoothstep(95.,130.,r);
  T0=mix(T0,vec4(0.,0.,0.,1.),far); T1*=1.-far;

  vec3 L=uSun;
  float sunVis=skyT;
  // ---- how much light is around
  vec3 skyAmb=textureLod(uEnv,vec2(.5,.35),6.).rgb*.75+textureLod(uEnv,vec2(.62,.12),5.).rgb*.25;

  // ---- dry to damp sand
  vec3 sandN=normalize(vec3(-T0.y,1.,-T0.z));
  float alb=.06*T0.a;
  // and water creeps a little way up through the sand from where it reached
  float wetA=sat(max(Wet,textureLod(uR1,uvW,5.2).g*(.25+.5*vnoise(q*.9))));
  // The tide is going out: all the way up to its last line the sand is saturated, no film on it but dull-wet, and it
  // takes the low sky over a wide angle. That, and not any light of its own, is what keeps black sand from going dead.
  float tide=0.,scum=0.;
  if(uHero>0.&&r<95.){
    float bed=R.r-R.g+.7*(vnoise(q*.11)-.5);
    tide=uHero*(1.-smoothstep(.55,2.3,bed+.2*(vnoise(q*.7)-.5)));
    // Every wave of the falling tide stopped somewhere. It left a line along the level it reached, a thread of spent
    // bubbles and grit; it is wetter below that line than above. Each is a contour of the beach, wandering a little.
    float s=(bed+.11*(vnoise(q*vec2(.31,.5))-.5)+.035*(vnoise(q*1.9)-.5))/.21;
    float f=fract(s),age=hash21(vec2(floor(s),3.));
    float wet0=1.-smoothstep(.25,.7,wetA);
    float keep=smoothstep(.34,.62,vnoise(q*.23+floor(s)*7.));
    tide*=mix(1.,mix(.72,1.12,f*f),wet0*keep*smoothstep(.05,.18,tide));
    scum=wet0*keep*tide*(.35+.65*age)*smoothstep(.052,.012,f)*smoothstep(.0,.006,f)*smoothstep(.3,.55,vnoise(q*vec2(6.,9.)));
  }
  float darken=mix(1.,.42,smoothstep(.02,.35,max(wetA,.6*tide)));
  // stays of ambient and the sun skimming over the ripples
  float ndl=max(dot(sandN,L),0.);
  float aoRip=.6+.52*smoothstep(-.1,.06,dot(sandN.xz,-L.xz));
  vec3 sand=alb*darken*(skyAmb*aoRip+uSunE*sunVis*ndl*.32)*mix(1.,mix(1.5,.7,T0.a*.6),T1.r);
  // the grains glance the low sun back at the camera: a broad silver sheen around the sun's azimuth
  vec3 Hv=normalize(V+L);
  vec2 sh=-Hv.xz/Hv.y;
  // still a film of water on it: nearly a mirror, and it does not leave the sand all at once
  float film=smoothstep(.62,.99,wetA+.2*(T0.a-.5)+.16*(vnoise(q*1.7)-.5)+.1*(vnoise(q*7.3)-.5));
  // and what is left standing lies in the troughs of the ripples
  film=max(film,tide*.8*smoothstep(.0006,-.0012,T0.x)*smoothstep(.4,.62,vnoise(q*.53)));
  float rough2=mix(mix(.1,.012,smoothstep(.1,.6,wetA)),.00035,film);
  rough2=mix(rough2,min(rough2,.034),tide);
  // the skerry out there is broken basalt, wet or not: nothing on it is smooth enough to mirror the sky
  float rocky=step(250.,r)*(1.-smoothstep(.06,.11,abs(th-uFigure.x)))*smoothstep(.2,.55,R.r-h);
  rough2=mix(rough2,.25,rocky);
  rough2=max(rough2+(T1.b+T1.a)*mix(1.,.35,film)+T1.r*.06,1e-4);
  vec2 ds=sh-T0.yz;
  float ex=dot(ds,ds)/(2.*rough2);
  float fcg=fresnel(dot(V,Hv));
  float P=exp(-ex)/(2.*PI*rough2);
  float zV=max(dot(V,sandN),.02);
  vec3 sunLobe=uSunE*sunVis*fcg*P/(4.*zV*Hv.y*Hv.y*Hv.y*Hv.y);
  // a damp film mirrors the sky through its roughness
  vec3 rS=reflect(d,sandN);
  float cS=max(dot(V,sandN),.02);
  float Fs=fresnel(cS);
  float rsg=sqrt(rough2);
  vec3 skyS=skyMirror(rS,rsg*.6+fw,rsg*1.9+fw);
  if(uHero>0.) skyS=mix(skyS,leadSea(skyS,d),.65);
  float mirror=max(mix(.22,1.,smoothstep(.2,.95,wetA))*smoothstep(.0,.2,wetA),.92*tide);
  // a pebble is a dull lump in it, not a mirror: what it shows of the sun and the sky goes with it
  float dull=1.-.96*sat(T1.r*2.);
  // the sheen off black grains is silver: it is the sky and the sun's white, there is no colour in the sand to tint it
  float sheenL=dot(sunLobe,vec3(.25,.65,.1));
  sunLobe=mix(sunLobe,vec3(sheenL)*vec3(.98,.99,1.03),.7*(1.-smoothstep(.3,.8,wetA)));
  vec3 sandSpec=(min(sunLobe*max(mix(.075,.55,smoothstep(.05,.5,wetA)),.1*tide),vec3(9.))+skyS*Fs*mirror*.92)*dull*(1.-.93*rocky);
  // fine mineral sparkle: isolated grains catching the sun
  float sp=0.;
  if(wa<.05){
    vec2 c=floor(vec2(th,rho)/(fw*1.2));
    float hs=hash21(c);
    float lobe=exp(-dot(sh,sh)/(2.*.11));
    float dens=.005+.022*lobe;
    if(hs<dens){
      vec2 f=fract(vec2(th,rho)/(fw*1.2))-.5;
      float b=hash21(c+3.3);
      sp=exp(-dot(f,f)*14.)*(.45+9.*lobe*b*b*b);
    }
  }
  sand+=sandSpec+vec3(.85,.9,1.)*sp*sunVis*uSunE*.012*(1.-.85*wetA);
  sand=mix(sand,skyAmb*.26+uSunE*sunVis*.012,scum*.8);

  // what a bubble raft is lit by: everything from everywhere, with no relief to catch it
  vec3 foamLit=vec3(.93,.96,1.)*(skyAmb*2.+uSunE*sunVis*.085);
  // ---- water: how deep and how it faces
  // the edge goes by a slightly wider look at the depth, or it follows the corners of the cells
  float hEff=mix(textureLod(uR0,uvW,1.4).g,h,smoothstep(25.,70.,r))-max(T0.r,0.)*.3;
  float px=min(wb,.2);
  float wetC=smoothstep(.0,.0025+px*.01,hEff-.0012);
  vec3 col=sand;
  vec3 N=sandN;
  if(wetC>0.){
    vec2 D=vec2(0.);
    vec2 s=sim;
    vec3 cov=vec3(0.);
    // sub-footprint roughness of the resolved flow
    float rough=length(sim)*.04;
    // short waves ride on the flow, and die in a thin film
    float shal=smoothstep(.006,.28,h);
    float m=shal;
    float along=wb;
    float g=gust(q,along);
    float gs=g*1.9;
    gs=gs/(1.+abs(gs))*1.4+.2*smoothstep(-.03,.05,g)-.1;
    float mg=exp(gs*.6);
    for(int i=0;i<NC;i++){
      float wgt=i<2?smoothstep(220.,430.,r):1.;
      wgt*=i>=2?m*mg:1.;
      if(wgt<=0.) continue;
      vec2 u=tileUV(i,q);
      // waves too short to span a few rows of pixels are roughness, not shape: left in the mean they comb the water
      vec2 tx=tileUV(i,nt*wa*1.),ty=tileUV(i,nr*wb*mix(1.,3.,uHero));
      vec4 b=textureGrad(uB,vec3(u,float(i)),tx,ty);
      float xz=textureGrad(uA,vec3(u,float(i)),tx,ty).a;
      s+=wgt*b.xy;
      cov+=wgt*wgt*vec3(max(b.z-b.x*b.x,0.),max(b.w-b.y*b.y,0.),xz-b.x*b.y);
    }
    cov+=vec3(.0012,.0012,0.)*m*mg*mg+vec3(rough*rough,rough*rough,0.);
    // a sheet running up the sand is no mirror: it takes the shape of the ripples under it, and it shivers
    if(uHero>0.&&r<70.){
      float sheet=uHero*(1.-smoothstep(.02,.16,h))*(1.-smoothstep(45.,70.,r));
      vec2 qs=vec2(dot(q,nt),dot(q,nr));
      float n1=vnoise(qs*vec2(1.3,3.4)+uT*vec2(.0,.9)),n2=vnoise(qs*vec2(4.1,11.)+vec2(3.1,uT*1.7));
      s+=sheet*(T0.yz*1.4+nr*(.085*(n1-.5)+.06*(n2-.5))+nt*.05*(vnoise(qs*vec2(3.,5.)+7.)-.5));
      cov+=sheet*vec3(.0009,.0016,0.);
    }
    cov.xy+=vec2(1.5e-5);
    N=normalize(vec3(-s.x,1.,-s.y));
    float zVw=dot(V,N);
    vec2 Vh=normalize(V.xz);
    vec2 Vp=vec2(-Vh.y,Vh.x);
    float sv2=max(dot(Vh,vec2(cov.x*Vh.x+cov.z*Vh.y,cov.z*Vh.x+cov.y*Vh.y)),1e-6);
    float sp2=max(dot(Vp,vec2(cov.x*Vp.x+cov.z*Vp.y,cov.z*Vp.x+cov.y*Vp.y)),1e-6);
    float sv=sqrt(sv2),sp_=sqrt(sp2);
    float tanE=max(zVw,.0005)/sqrt(max(1.-zVw*zVw,1e-5));
    float tt=tanE/sv;
    float Ph=PhiN(tt),ph=phiN(tt);
    float mu=Ph/max(tt*Ph+ph,1e-4);
    float spreadV=mix(.66,1.,smoothstep(0.,2.5,tt));
    vec3 refl=vec3(0.); float Fm=0.;
    const float ZV[4]=float[4](-1.05,-.35,.35,1.05);
    const float ZP[4]=float[4](.75,-.75,.75,-.75);
    float spread=2.*sqrt(sv2*spreadV*spreadV*.35+sp2*.55);
    float lod=max(log2(spread/.00307),0.);
    if(spread<.05){
      // a thin smooth sheet is a mirror: the sky at its own resolution
      vec2 sk=s-Vh*mu*sv;
      vec3 n=normalize(vec3(-sk.x,1.,-sk.y));
      float c=max(dot(V,n),.012);
      Fm=fresnel(c);
      refl=Fm*skyMirror(2.*c*n-V,sp_*2.+fw,sv*2.+fw);
    } else {
      for(int k=0;k<4;k++){
        vec2 sk=s-Vh*(mu+ZV[k]*spreadV)*sv+Vp*ZP[k]*sp_;
        vec3 n=normalize(vec3(-sk.x,1.,-sk.y));
        float c=max(dot(V,n),.012);
        float F=fresnel(c);
        vec3 rr=2.*c*n-V;
        refl+=F*envAt(rr,lod);
        Fm+=F;
      }
      refl*=.25; Fm*=.25;
    }
    refl=leadSea(refl,d);
    // sun on the water: the slope distribution against the half vector
    vec3 glit=vec3(0.);
    vec2 dsw=sh-s;
    // the waves between the solver's and the pixel's tilt whole patches of water out of the sun's way, which is
    // what breaks a path up. Each length runs at its own speed.
    if(uHero>0.){
      vec2 qw=vec2(dot(q,vec2(.985,-.17)),dot(q,vec2(.17,.985)));
      float tilt=0.,lam=2.1;
      for(int k=0;k<6;k++){
        float wg=smoothstep(.45,1.5,lam/wb)*(1.-smoothstep(9.,22.,lam/wb));
        if(wg>0.){
          float c=sqrt(1.56*lam);
          tilt+=wg*(vnoise(vec2(qw.x/(lam*2.7),(qw.y+uT*c)/lam)+float(k)*7.3)-.5);
        }
        lam*=2.45;
      }
      dsw-=nr*tilt*.11*uHero*(.4+.6*m);
    }
    const float sunV=2.5e-6;
    float a=cov.x+sunV,bb=cov.y+sunV,cc=cov.z;
    float det=max(a*bb-cc*cc,1e-9);
    float exw=(bb*dsw.x*dsw.x-2.*cc*dsw.x*dsw.y+a*dsw.y*dsw.y)/det;
    if(exw<140.){
      const float K2=2.1;
      float Pw=(.945*exp(-.5*exw)+.055*exp(-.5*exw/(K2*K2))/(K2*K2))/(2.*PI*sqrt(det));
      vec2 Lh=normalize(L.xz);
      float sl2=max(dot(Lh,vec2(cov.x*Lh.x+cov.z*Lh.y,cov.z*Lh.x+cov.y*Lh.y)),1e-6);
      float zL=max(dot(L,N),.01);
      float zVc=max(zVw,.01);
      float zH=Hv.y;
      float F=fresnel(dot(V,Hv));
      float Lmask=1.+lambdaS(zVc,sv2)+lambdaS(zL,sl2);
      glit=uSunE*sunVis*F*Pw/(4.*zVc*zH*zH*zH*zH*Lmask);
      glit=min(glit,vec3(400.));
      float cellg=.00095*pow(2.,max(ceil(log2(fw*1.05/.00095)),0.));
      float Acell=dist*dist*cellg*cellg/gr;
      const float Af=4.5e-4;
      float Lone=dot(uSunL,vec3(.2126,.7152,.0722))*sunVis*F*Af/Acell;
      float Lmean=dot(glit,vec3(.2126,.7152,.0722));
      float Ncell=Lmean/max(Lone,1e-4);
      float cap=mix(2.5,70.,smoothstep(.02,.6,Ncell));
      if(uMode!=2){
        float sO=uHero<1.?sparkle(d,Ncell,cellg):1.;
        float sG=uHero>0.?glints(d,Ncell,mix(.75,1.25,smoothstep(.02,.09,sp_)),fw):1.;
        glit=min(glit*mix(sO,sG,uHero),vec3(cap,cap,cap*.67));
      }
    }
    // what comes up from under the surface: suspended sand, the bed seen through a thin sheet
    vec3 bodyDeep=vec3(.0042,.0085,.0098)*skyAmb*2.2;
    float vis=exp(-h*5.5);
    vec3 body=mix(bodyDeep,sand*.82,vis);
    // the top of a steep face lets the low sun through
    float faces=sat(dot(N.xz,Vh)*3.);
    float back=pow(sat(dot(d,normalize(vec3(L.x,0.,L.z)))),3.);
    vec3 sss=vec3(.13,.26,.24)*uSunE*sunVis*.05*faces*back*smoothstep(.1,1.2,h)*(.3+1.1*smoothstep(.04,.5,B.a));
    // water full of bubbles scatters whatever reaches it: milky, under the shine of its surface
    float milk=(1.-exp(-2.4*Fo))*smoothstep(.0,.05,h);
    // in a thin sheet over black sand there is no depth for the bubbles to be in: it stays clear between the strands
    if(uHero>0.) milk*=mix(1.,smoothstep(.12,.8,h),uHero*(1.-smoothstep(50.,110.,r)));
    body=mix(body,foamLit*.42,milk);
    glit*=1.-.75*milk;
    vec3 w=(body+sss)*(1.-Fm)+refl+glit;
    col=mix(sand,w,wetC);
  }

  // ---- foam
  float amt; vec4 rel;
  // a sheet a few centimetres thin cannot hold a raft of foam: it is spread out into lace and clear water
  float hF=r<45.?mix(Rs.g,h,smoothstep(25.,45.,r)):h;
  float Fv=Fo*.88*mix(.42,1.,smoothstep(.015,.15,hF));
  // and whatever is running up the beach carries a scum of bubbles with it, thickest at its edge
  if(uHero>0.&&r<70.) Fv=max(Fv,uHero*wetC*(1.-smoothstep(.07,.2,hF))*(1.-smoothstep(45.,70.,r))*(.26+.2*vnoise(q*.73)));
  // its footprint goes by the wave it rides, not by every lump on it
  float slF=max(dot(sim,nr),0.);
  const float FACE=1.6;
  float wbF=wa*(1.+FACE*slF)/(gr+slF);
  // and it lies along the surface, not along its plan: up a steep face the height does the moving,
  // or the lace would be drawn out into a curtain over the few centimetres of plan the face covers
  float eF=r<45.?mix(Rs.r,R.r,smoothstep(25.,45.,r)):R.r;
  vec2 qF=q+nr*FACE*(eF-textureLod(uR0,uv0,4.5).r);
  // the two sets of coordinates: patch by patch the lace hands over from one to the other, which is
  // also what lace does on its own as its bubbles burst and its strands draw together
  float nsw=.12+.76*vnoise(q/max(1.3,wa/gr*2.2));
  float sw=smoothstep(nsw-.12,nsw+.12,uMatBlend);
  float lace;
  if(sw<=.001) lace=foamLace(qF+dm.xy,Fv,nt*wa,nr*wbF,amt,rel);
  else if(sw>=.999) lace=foamLace(qF+dm.zw,Fv,nt*wa,nr*wbF,amt,rel);
  else {
    float a2; vec4 r2;
    float l1=foamLace(qF+dm.xy,Fv,nt*wa,nr*wbF,amt,rel);
    float l2=foamLace(qF+dm.zw,Fv,nt*wa,nr*wbF,a2,r2);
    lace=mix(l1,l2,sw); amt=mix(amt,a2,sw); rel=mix(rel,r2,sw);
  }
  lace=mix(lace,amt,smoothstep(.3,1.4,wbF));
  if(lace>0.){
    // matte: thousands of small bubbles scatter the light in all directions, so a raft has tone but no
    // shine and no modelling. It is a little darker where thin and on the shaded face of a roller
    float thick=smoothstep(.1,.8,amt);
    // even a solid raft is cells: what the strands do at the edge of a patch, the tone does inside it
    float ao=mix(.8,1.03,thick)*mix(.92,1.05,rel.a)*mix(.84,1.03,smoothstep(.25,.75,rel.b))*mix(.88,1.04,smoothstep(.2,.8,rel.g));
    float faceDark=mix(1.,.62,sat(dot(sim,nr)*1.6));
    vec3 Ec=skyAmb*2.*ao*faceDark;
    float toSun=pow(sat(dot(d,normalize(vec3(L.x,0.,L.z)))),6.);
    vec3 Es=uSunE*sunVis*(.085+.2*toSun*thick)*mix(.75,1.,thick)*mix(.7,1.,faceDark);
    if(uHero>0.){
      vec2 he=sw<.5?foamHeaps(qF+dm.xy,nt*wa,nr*wbF):foamHeaps(qF+dm.zw,nt*wa,nr*wbF);
      if(sw>.02&&sw<.98) he=mix(foamHeaps(qF+dm.xy,nt*wa,nr*wbF),foamHeaps(qF+dm.zw,nt*wa,nr*wbF),sw);
      // close to, the lace and the cells are the structure: heaps there only shade it a little
      float d0=smoothstep(.2,.75,amt)*uHero*mix(.3,1.,smoothstep(26.,70.,r));
      float lit=sat(.5+7.*he.y);
      float heap=smoothstep(.3,.7,he.x);
      Ec*=mix(1.,mix(.74,1.08,heap)*mix(.9,1.06,lit),d0);
      Es*=mix(1.,mix(.1,1.9,lit)*mix(.55,1.2,heap),d0);
    }
    vec3 foamCol=vec3(.93,.96,1.)*(Ec+Es);
    // the last thin threads are a few bubbles wide: they let the water through
    col=mix(col,foamCol,lace*mix(.55,.97,smoothstep(.08,.45,amt)));
  }

  // ---- the air between here and the camera
  float km=dist*.001;
  vec3 Ta=exp(-uAir*km*vec3(.86,1.,1.22));
  vec2 pa=vec2(atan(d.x,d.z)/(2.*PI)+.5,.04);
  vec3 air=texture(uBg,pa).rgb;
  return col*Ta+air*(1.-Ta);
}

// ------------------------------------------------------------------ everything for one direction
// the surf is only worked out over part of the circle. a drop sees all of it: past the ends the beach just goes on
float foldAz(float th){
  float lo=uPol.x+.002,hi=uPol.x+uPol.y*float(uNS.x)-.002;
  if(th<lo) th=2.*lo-th;
  if(th>hi) th=2.*hi-th;
  return clamp(th,lo,hi);
}
vec3 shade(vec3 d,vec3 dx,vec3 dy,float fw){
  float c2=1.-d.y*d.y;
  float dyH=-sqrt(4.*c2*uCurv*uH);       // horizon
  float sd=(dyH-d.y)/fw;                 // >0 below the horizon, in pixels
  float seaA=sat(sd+.5);
  vec3 col=vec3(0.);
  {
    float T;
    vec3 ds=normalize(uSun);
    skyColor(ds,vec3(0.),vec3(0.),T);
    skyT=T;
  }
  if(seaA<1.){
    float T;
    vec3 ds=d; ds.y=max(d.y,dyH+fw*.25);
    vec3 sky=skyColor(normalize(ds),dx,dy,T);
    sky+=sunDisc(d,fw,T);
    float bc=birds(d,fw);
    vec3 birdC=sky*.085+vec3(.012,.009,.009);
    sky=mix(sky,birdC,bc);
    col=sky;
  }
  float dist=0.;
  float rhoAir=-1.;
  if(seaA>0.){
    vec3 ds=d; ds.y=min(d.y,dyH-fw*.25);
    ds=normalize(ds);
    float th=atan(ds.x,ds.z);
    float tanEl=ds.y/sqrt(max(1.-ds.y*ds.y,1e-6));
    float rhoHit;
    vec3 sea;
    float thc=foldAz(th);
    if(marchBeach(thc,tanEl,fw,rhoHit)){ sea=shadeBeach(ds,dx,dy,fw,thc,rhoHit,tanEl,dist); rhoAir=rhoHit; }
    else sea=waterColor(ds,dx,dy,fw,dist);
    float bc=birds(d,fw);
    if(bc>0.) sea=mix(sea,sea*.1+vec3(.006),bc);
    col=mix(col,sea,seaA);
  }
  float az=atan(d.x,d.z),el=asin(d.y);
  {
    vec3 d1=normalize(d+dx),d2=normalize(d+dy);
    vec2 g1=vec2(atan(d1.x,d1.z)-az,asin(d1.y)-el),g2=vec2(atan(d2.x,d2.z)-az,asin(d2.y)-el);
    vec4 st=stackAt(az,el,g1,g2);
    col=col*(1.-st.a)+st.rgb;
  }
  // salt over the surf, between here and everything: it belongs to the air, so it crosses the horizon
  if(el<.03&&uHero<1.){
    float thc=foldAz(az);
    float tanE=d.y/sqrt(max(1.-d.y*d.y,1e-6));
    float rhoE=rhoAir>0.?min(rhoAir,log(uRmax)):log(uRmax);
    float tauS=sprayAlong(thc,tanE,rhoE)*.05*(1.-smoothstep(.0,.03,el))*(1.-uHero);
    if(tauS>.0005){
      float toS=dot(d,uSun);
      float wisp=.72+.56*vnoise(vec2(thc*90.+uT*.11,rhoE*26.-uT*.17));
      float ph=.16+1.5*pow(sat(toS),24.)+.5*pow(sat(toS),6.);
      vec3 skyAmb=textureLod(uEnv,vec2(.5,.35),6.).rgb*.75+textureLod(uEnv,vec2(.62,.12),5.).rgb*.25;
      vec3 sprayC=skyAmb*.75+uSunE*skyT*ph*.32;
      col=mix(col,sprayC,1.-exp(-tauS*wisp));
    }
  }
  float cov=figure(d,fw);
  if(cov>0.){
    vec2 pa=vec2(atan(d.x,d.z)/(2.*PI)+.5,.05);
    vec3 air=texture(uBg,pa).rgb;
    vec3 rock=vec3(.012,.0115,.015);
    const vec3 Tr=vec3(.94,.94,.93);
    col=mix(col,rock*Tr+air*(1.-Tr),cov);
  }
  return col;
}

