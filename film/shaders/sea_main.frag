#version 410
// The beach. One function of a view direction: sky, sun, birds, stacks, then along the azimuth of
// that direction, whatever the ray first meets: the simulated water and the sand.
in vec2 uv; out vec4 o;
#include "common.glsl"
const float PI=3.14159265;
const int NC=5;

uniform float uAspect,uTan,uT,uH,uCurv,uHmaxO,uAir,uGain,uHorEl;
uniform int uMode;              // 0 normal, 1 through a water bead, 2 equirect blocks for the streaming world
uniform vec3 uF,uR,uU;
uniform vec4 uDrop;             // cx,cy,R,lens
uniform vec2 uEq;               // edge_az, horizon el of the block grid
uniform vec4 uCas[NC];          // cos/L, sin/L, texel size, unused
uniform sampler2DArray uA,uB,uCloud,uDome;
uniform vec4 uSkyWin;            // az0,az1,el0,el1 of the high-resolution window of the sky
uniform sampler2D uBg,uEnv,uBlocks;
uniform vec3 uSun,uSunE,uSunL;
uniform vec3 uDrift[3];
uniform vec4 uBird[24];         // az,el,half span,wing lift
uniform float uMatLag;          // seconds the material coordinates are behind the water
uniform float uHero;            // 1 looking at the sun, 0 on the side of the beach: that view is left exactly as it is
uniform float uFoot;            // how much of the pixel each look of a frame has to answer for
uniform vec4 uFigure;           // az, el of the feet, height, unused
uniform sampler2D uR0,uR1,uHmax,uMat,uT0,uT1,uLace,uLaceN,uStacks;
uniform vec4 uLaceTh;
uniform float uLaceL;           // metres of the strands' tile
uniform vec4 uStackWin;
uniform vec4 uPol;              // th0,dth,rho0,drho of the water grid
uniform ivec2 uNS;
uniform vec4 uTer;              // th0,th1,rho0,rho1 of the sand bake
uniform float uRmax,uBlock,uTop,uMatBlend;
uniform vec4 uCrack;            // on, t_arrive, edge_az
uniform vec2 uTile;
uniform ivec2 uGrid;
uniform int uSub;

float sat(float x){ return clamp(x,0.,1.); }

// ------------------------------------------------------------------ view
vec3 viewDir(vec2 v){
  if(uMode==2){
    vec2 g=uv*vec2(uGrid)-vec2(0.,float(uGrid.y)*.5);
    float az=uEq.x+g.x*uTile.x, el=uEq.y+g.y*uTile.y;
    return vec3(sin(az)*cos(el),sin(el),cos(az)*cos(el));
  }
  if(uMode==1){
    vec2 rho=(v-uDrop.xy)/uDrop.z;
    float g=mix(1.,1./sqrt(max(1.02-dot(rho,rho),.02)),uDrop.w);
    // .27*3.09: at R=3.09, lens=0 this is exactly the normal mapping at roll=pi, tan_half=.27
    const float K=.8343;
    return normalize(uF-rho.x*g*K*uR-rho.y*g*K*uU);
  }
  return normalize(uF+v.x*uTan*uR+v.y*uTan*uU);
}

// ------------------------------------------------------------------ sky
vec2 panoUV(vec3 d){
  float az=atan(d.x,d.z);
  float el=asin(clamp(d.y,0.,1.));
  return vec2(az/(2.*PI)+.5,sqrt(el/(.5*PI)));
}
vec2 wrapDu(vec2 g){ g.x-=round(g.x); return g; }

// the sky's texels are a little larger than the pixels, and linear interpolation beats against the frame
// in a fine mesh, keeping what noise the bake left at texel centres and averaging it between them
vec4 cloudTap(vec2 w,float k,vec2 wx,vec2 wy){
  vec2 n=vec2(textureSize(uCloud,0).xy);
  vec2 g=w*n-.5,f=fract(g); g-=f;
  vec2 f2=f*f,f3=f2*f,c=1.-f;
  vec2 w0=c*c*c/6.,w1=(4.-6.*f2+3.*f3)/6.,w3=f3/6.,w2=1.-w0-w1-w3;
  vec2 a=w0+w1,b=w2+w3;
  vec2 p0=(g-.5+w1/a)/n,p1=(g+1.5+w3/b)/n;
  return a.y*(a.x*textureGrad(uCloud,vec3(p0,k),wx,wy)+b.x*textureGrad(uCloud,vec3(p1.x,p0.y,k),wx,wy))
        +b.y*(a.x*textureGrad(uCloud,vec3(p0.x,p1.y,k),wx,wy)+b.x*textureGrad(uCloud,vec3(p1,k),wx,wy));
}

// clear sky + clouds; T is what the clouds let through
vec3 skyColor(vec3 d,vec3 dx,vec3 dy,out float T){
  vec2 p=panoUV(d);
  vec3 acc=vec3(0.); T=1.;
  float az0=atan(d.x,d.z),el0=asin(clamp(d.y,0.,1.));
  // the camera's window of the sky is baked at the pixel; the rest of the dome much coarser
  float inW=smoothstep(uSkyWin.x,uSkyWin.x+.03,az0)*(1.-smoothstep(uSkyWin.y-.03,uSkyWin.y,az0))*(1.-smoothstep(uSkyWin.w-.03,uSkyWin.w-.002,el0));
  for(int k=0;k<3;k++){
    vec3 dk=normalize(d+uDrift[k]*max(d.y,0.));
    vec4 c;
    vec2 u=panoUV(dk);
    if(inW<1.){
      vec2 gx=wrapDu(panoUV(normalize(dk+dx))-u),gy=wrapDu(panoUV(normalize(dk+dy))-u);
      c=textureGrad(uDome,vec3(u,float(k)),gx,gy);
    } else c=vec4(0.);
    if(inW>0.){
      float az=atan(dk.x,dk.z),el=asin(clamp(dk.y,0.,1.));
      vec2 w=vec2((az-uSkyWin.x)/(uSkyWin.y-uSkyWin.x),(el-uSkyWin.z)/(uSkyWin.w-uSkyWin.z));
      vec3 dx2=normalize(dk+dx),dy2=normalize(dk+dy);
      vec2 wx=vec2((atan(dx2.x,dx2.z)-az)/(uSkyWin.y-uSkyWin.x),(asin(clamp(dx2.y,0.,1.))-el)/(uSkyWin.w-uSkyWin.z));
      vec2 wy=vec2((atan(dy2.x,dy2.z)-az)/(uSkyWin.y-uSkyWin.x),(asin(clamp(dy2.y,0.,1.))-el)/(uSkyWin.w-uSkyWin.z));
      c=mix(c,cloudTap(w,float(k),wx,wy),inW);
    }
    acc+=T*c.rgb; T*=c.a;
  }
  return acc+T*texture(uBg,p).rgb;
}

vec3 sunDisc(vec3 d,float fw,float T){
  vec3 q=d-uSun;
  float th=length(vec3(q.x,q.y*1.04,q.z));     // refraction flattens it a little this low
  const float rs=.00466;
  // the aureole: large haze particles scatter in a narrow forward cone, falling off as 1/theta^2
  float tha=max(th,fw*.7);
  vec3 aur=uSunE*vec3(1.,.985,.95)*(.00062/(tha*tha+.00003))*(.25+.75*T);
  aur*=1.-smoothstep(.12,.5,th);
  if(th>rs+3.*fw+.002) return aur;
  float w=max(fw*.8,3e-5);
  float conserve=min(1.,rs*rs/(w*w));
  float cover=1.-smoothstep(max(rs-w,0.),rs+w,th);
  float r=min(th/rs,1.);
  float limb=mix(.55,1.,sqrt(1.-r*r));
  return aur+uSunL*limb*cover*conserve*T;
}

// ------------------------------------------------------------------ birds
float sdSeg(vec2 p,vec2 a,vec2 b){
  vec2 pa=p-a,ba=b-a; float h=sat(dot(pa,ba)/dot(ba,ba));
  return length(pa-ba*h);
}
float birds(vec3 d,float fw){
  float az=atan(d.x,d.z),el=asin(d.y),ce=cos(el);
  float cov=0.;
  for(int i=0;i<24;i++){
    vec4 b=uBird[i];
    vec2 q=vec2((az-b.x)*ce,el-b.y);
    float s=b.z;
    if(abs(q.x)>s*1.5+fw*2.||abs(q.y)>s*1.3+fw*2.) continue;
    vec2 p=q/s;
    p.x=abs(p.x);
    float lift=b.w;
    float bend=lift*.42+.09;
    float dd=min(sdSeg(p,vec2(0.),vec2(.46,bend))-.085,sdSeg(p,vec2(.46,bend),vec2(1.,lift*.95-.05))-.04);
    dd=min(dd,length(p*vec2(2.6,6.))-.42);
    float w=fw/s;
    float c=1.-smoothstep(-w,w,dd-.3*w);
    // below a pixel a bird keeps its ink, not its shape
    c*=min(1.,(s*1.6)/max(fw,1e-6));
    cov=max(cov,c);
  }
  return cov;
}

float boxSd(vec2 p,vec2 c,vec2 h){ vec2 q=abs(p-c)-h; return length(max(q,0.))+min(max(q.x,q.y),0.); }
// a person, standing still, holding a phone up at the camera
float figure(vec3 d,float fw){
  float az=atan(d.x,d.z),el=asin(d.y);
  vec2 p=vec2((az-uFigure.x)*cos(el),el-uFigure.y)/uFigure.z;
  if(abs(p.x)>1.2||p.y<-.3||p.y>1.6) return 0.;
  float w=fw/uFigure.z;
  float sd=1e3;
  sd=min(sd,sdSeg(p,vec2(-.04,.0),vec2(-.025,.46))-.058);    // legs
  sd=min(sd,sdSeg(p,vec2(.04,.0),vec2(.025,.46))-.058);
  sd=min(sd,sdSeg(p,vec2(.0,.47),vec2(.0,.75))-.1);          // torso
  sd=min(sd,length(p-vec2(.0,.91))-.082);                    // head
  sd=min(sd,sdSeg(p,vec2(-.08,.74),vec2(-.12,.46))-.04);     // hanging arm
  sd=min(sd,sdSeg(p,vec2(.08,.77),vec2(.235,.98))-.045);     // raised arm
  sd=min(sd,boxSd(p,vec2(.285,1.07),vec2(.078,.125)));       // the rectangle
  return 1.-smoothstep(-w*.7,w*.7,sd);
}

// ------------------------------------------------------------------ water
vec2 tileUV(int i,vec2 q){ vec4 c=uCas[i]; return vec2(c.x*q.x+c.y*q.y,-c.y*q.x+c.x*q.y); }

float rayY(float t,vec3 d){ return uH+t*d.y-t*t*(1.-d.y*d.y)*uCurv; }
float planeT(vec3 d,float h){
  float c2=1.-d.y*d.y;
  float dyH=-sqrt(4.*c2*uCurv*h);
  float dy=min(d.y,dyH-2e-6);
  return 2.*h/(-dy+sqrt(max(dy*dy-4.*c2*uCurv*h,0.)));
}

float fresnel(float c){
  c=clamp(c,0.,1.);
  const float n2=1.7689;
  float g=sqrt(n2-1.+c*c);
  float a=(g-c)/(g+c);
  float b=(c*(g+c)-1.)/(c*(g-c)+1.);
  return .5*a*a*(1.+b*b);
}
float erfcA(float x){ return 2.*exp(-x*x)/(2.319*x+sqrt(4.+1.52*x*x)); }
float lambdaS(float cosT,float s2){
  float v=cosT/sqrt(max((1.-cosT*cosT)*2.*s2,1e-7));
  return max(0.,(exp(-v*v)-v*sqrt(PI)*erfcA(v))/(2.*v*sqrt(PI)));
}
float phiN(float t){ return exp(-.5*t*t)/2.5066; }
float PhiN(float t){ return .5*erfcA(-t*.7071*step(t,0.))*step(t,0.)+step(0.,t)*(1.-.5*erfcA(t*.7071)); }

// waves' height along the ray: two biggest cascades, with their horizontal chop inverted
float heightAt(vec2 q,float l0,float l1,float w){
  vec4 a=textureLod(uA,vec3(tileUV(0,q),0.),l0);
  vec4 b=textureLod(uA,vec3(tileUV(1,q),1.),l1);
  float h0=textureLod(uA,vec3(tileUV(0,q-a.xz),0.),l0).g;
  float h1=textureLod(uA,vec3(tileUV(1,q-b.xz),1.),l1).g;
  return (h0+h1)*w;
}

// roughness patches: gusts and slicks on the water; rotation-invariant, never repeating
float gust(vec2 q,float along){
  vec2 w=vec2(.906*q.x-.423*q.y,.423*q.x+.906*q.y);
  vec2 drift=vec2(0.,uT*3.);
  float s=0.,a=.5,tot=0.;
  // the edges of a patch wander, so seen at grazing angles they never become straight lines
  w+=vec2(vnoise(w/410.+vec2(3.7,8.1)),vnoise(w/370.+vec2(9.2,1.3)))*vec2(520.,380.)
    +vec2(vnoise(w/97.+vec2(1.1,5.9)),vnoise(w/89.+vec2(6.3,2.7)))*vec2(70.,60.);
  vec2 p=w*vec2(1./1100.,1./430.)+vec2(11.3,4.1);
  float f=1./430.;
  for(int i=0;i<4;i++){
    float fade=1.-smoothstep(.35,1.4,along*f);
    s+=a*fade*(vnoise(p+drift*f)-.5); tot+=a;
    p=mat2(1.7,1.1,-1.1,1.7)*p+vec2(3.1,1.7); a*=.62; f*=2.03;
  }
  return s/tot;
}

// sparkles: a sun glint is one facet in a pixel. The mean is the analytic glitter; what a real
// lens sees is that mean counted out in discrete, short-lived glints.
float sparkAmp(vec2 k,float N){
  if(N<1.) return hash21(k)<N?1./N:0.;
  return max(1.+gauss(k).x/sqrt(N),0.);
}
float sparkle(vec3 d,float Ncell,float cell){
  if(Ncell>64.) return 1.;
  vec2 g=vec2(atan(d.x,d.z),asin(d.y))/cell;
  vec2 c0=floor(g);
  float rho=mix(.3,.52,smoothstep(.25,1.5,Ncell));   // a lone glint is a point, a crowd is a texture
  float S=0.;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 id=c0+vec2(float(i),float(j));
    float rate=1./(.085+.075*hash21(id+.7));
    float tt=uT*rate+hash21(id*1.73+3.1);
    float s0=floor(tt),fr=tt-s0;
    float e0=sin(PI*fr); e0*=e0;
    float s1=floor(tt+.5);
    float a=sparkAmp(id+vec2(s0*1.37,s0*.73),Ncell)*e0+sparkAmp(id+vec2(s1*1.37,s1*.73)+31.,Ncell)*(1.-e0);
    if(a<=0.) continue;
    vec2 ctr=id+.5+.9*(hash22(id+9.1)-.5);
    vec2 r=g-ctr;
    S+=a*exp(-dot(r,r)/(2.*rho*rho));
  }
  return S/(2.*PI*rho*rho);
}

// What the lens really gets from a low sun: a facet of a wave, turned to the sun for a moment, wider than it
// is deep and foreshortened further, so a sliver lying along the wave and never a dot. N facets to a cell
// keep the mean: more of them fill more of the cells, and the path closes up only where there are very many.
float glints(vec3 d,float N,float sz,float fw){
  float az=atan(d.x,d.z),el=asin(d.y);
  float ch=fw/uFoot*1.7;
  float row0=floor(el/ch);
  float fill=min(N*.8/(1.+.4*N),.94);
  if(fill<1e-6) return 0.;
  float grow=mix(1.,1.65,smoothstep(6.,300.,N));
  float S=0.;
  for(int j=-1;j<=1;j++){
    float row=row0+float(j);
    float rr=uH/max(-(row+.5)*ch,.004);
    float cw=clamp(.13/rr,ch*2.6,ch*6.5);
    float gx=az/cw+hash11(row*.37+.11)*7.;
    float c0=floor(gx);
    for(int i=-1;i<=1;i++){
      vec2 id=vec2(c0+float(i),row);
      float tt=uT/(.05+.075*hash21(id+.7))+hash21(id*1.73+3.1);
      float s0=floor(tt),fr=tt-s0;
      float e0=sin(PI*fr); e0*=e0;
      float s1=floor(tt+.5);
      float a=(hash21(id+vec2(s0*1.37,s0*.73))<fill?e0:0.)+(hash21(id+vec2(s1*1.37,s1*.73)+31.)<fill?1.-e0:0.);
      if(a<=0.) continue;
      vec2 hs=hash22(id+9.1),hl=hash22(id+4.7);
      vec2 p=vec2(gx,el/ch)-(id+.5+.72*(hs-.5));
      // not one is straight or level: each leans with its bit of wave and bows a little over it
      p.y+=(hl.y-.5)*.5*p.x+(hl.x-.5)*.9*p.x*p.x;
      float lx=mix(.26,.66,hl.x*hl.x)*grow*sz,ly=.46*grow;
      vec2 u=p/vec2(lx,ly*(1.-.5*min(p.x*p.x/(lx*lx),1.)));
      float q=dot(u,u);
      if(q<1.){ q=1.-q; S+=a*q*q*3./(PI*.43*.46*grow*grow*sz); }
    }
  }
  return S/fill;
}

vec3 envAt(vec3 r,float lod){
  float az=atan(r.x,r.z);
  float el=asin(clamp(r.y,.0005,1.));
  return textureLod(uEnv,vec2(az/(2.*PI)+.5,el/(.5*PI)),lod).rgb;
}

// Facing a low sun the sea is not warm. What a wave turns to the camera is tilted to the sky well above the glow, and
// the glow's colour is lost over the spread of the facets: lead and silver, warm only in the path itself.
vec3 leadSea(vec3 refl,vec3 d){
  if(uHero<=0.) return refl;
  float l=dot(refl,vec3(.25,.65,.1));
  vec3 c=l/(1.+.42*l)*vec3(.9,.965,1.13);
  return mix(refl,c,uHero*mix(.6,.9,smoothstep(.01,.16,abs(atan(d.x,d.z)))));
}

vec3 waterColor(vec3 d,vec3 dx,vec3 dy,float fw,out float dist){
  vec3 V=-d;
  // plane hit and its screen footprint (safe in any control flow: finite differences of the direction)
  float tp=planeT(d,uH);
  vec2 q0=d.xz*tp;
  vec3 ddx=normalize(d+dx),ddy=normalize(d+dy);
  vec2 dqx=ddx.xz*planeT(ddx,uH)-q0,dqy=ddy.xz*planeT(ddy,uH)-q0;
  float wa=max(tp*fw,.004);
  float grazing=max(-d.y,.0012);
  float wb=wa/grazing;
  float tHit=tp;
  vec2 hitQ=q0;
  float hgt=0.;
  // raymarch the two big cascades while they are resolved
  float geo=sqrt(wa*wb);
  float l0=max(log2(geo/uCas[0].z),0.),l1=max(log2(geo/uCas[1].z),0.);
  float mw=1.-smoothstep(.8,1.7,l0);     // beyond that the waves are statistics, not geometry
  if(mw>0.){
    float hm=uHmaxO*mw;
    float t0=planeT(d,max(uH-hm,1.)),t1=planeT(d,uH+hm);
    const int N=16;
    float ta=t0,ya=rayY(ta,d)-heightAt(d.xz*ta,l0,l1,mw),tb=ta,yb=ya;
    bool hit=false;
    if(ya<=0.){ tb=ta; }
    else for(int i=1;i<=N;i++){
      tb=t0+(t1-t0)*float(i)/float(N);
      yb=rayY(tb,d)-heightAt(d.xz*tb,l0,l1,mw);
      if(yb<=0.){ hit=true; break; }
      ta=tb; ya=yb;
    }
    if(ya<=0.) tHit=t0;
    else if(hit){
      for(int i=0;i<3;i++){
        float tm=ta+(tb-ta)*ya/max(ya-yb,1e-5);
        float ym=rayY(tm,d)-heightAt(d.xz*tm,l0,l1,mw);
        if(ym>0.){ ta=tm; ya=ym; } else { tb=tm; yb=ym; }
      }
      tHit=ta+(tb-ta)*ya/max(ya-yb,1e-5);
    }
    hitQ=d.xz*tHit;
  }
  dist=tHit;
  // displacement of the riding waves
  vec2 D=vec2(0.);
  vec2 s=vec2(0.); vec3 cov=vec3(0.);
  float along=wb;
  float g=gust(hitQ,along);
  // patches have edges: a gust is a place, not a gradient
  float gs=g*1.9;
  gs=gs/(1.+abs(gs))*1.4+.2*smoothstep(-.03,.05,g)-.1;
  float mg=exp(gs*mix(.9,.55,smoothstep(40.,260.,wb)));        // patch roughness
  for(int i=0;i<NC;i++){
    vec2 Q=hitQ;
    if(i<3){
      vec4 a0=textureLod(uA,vec3(tileUV(i,hitQ),float(i)),max(log2(wa/uCas[i].z),0.));
      Q=hitQ-a0.xz;
      D+=a0.xz;
      hgt+=textureLod(uA,vec3(tileUV(i,Q),float(i)),max(log2(wa/uCas[i].z),0.)).g;
    } else Q=hitQ-D;
    vec2 u=tileUV(i,Q);
    vec2 gx=tileUV(i,dqx),gy=tileUV(i,dqy);
    vec4 b=textureGrad(uB,vec3(u,float(i)),gx,gy);
    float xz=textureGrad(uA,vec3(u,float(i)),gx,gy).a;
    float m=i>=2?mg:1.;
    s+=m*b.xy;
    cov+=m*m*vec3(max(b.z-b.x*b.x,0.),max(b.w-b.y*b.y,0.),xz-b.x*b.y);
  }
  // waves shorter than the last cascade, and a pixel's own blur of the sun
  cov+=vec3(.0016,.0016,0.)*mg*mg;
  // far out the swell runs at the camera: little of its slope is sideways, and the path narrows
  cov.x*=mix(1.,.38,smoothstep(300.,2600.,tHit));

  vec3 N=normalize(vec3(-s.x,1.,-s.y));
  float zV=dot(V,N);
  vec2 Vh=normalize(V.xz);
  vec2 Vp=vec2(-Vh.y,Vh.x);
  float sv2=max(dot(Vh,vec2(cov.x*Vh.x+cov.z*Vh.y,cov.z*Vh.x+cov.y*Vh.y)),1e-6);
  float sp2=max(dot(Vp,vec2(cov.x*Vp.x+cov.z*Vp.y,cov.z*Vp.x+cov.y*Vp.y)),1e-6);
  float sv=sqrt(sv2),sp=sqrt(sp2);
  // only the facets that face the camera are visible: mean and spread of the visible slopes
  float tanE=max(zV,.0005)/sqrt(max(1.-zV*zV,1e-5));
  float tt=tanE/sv;
  float Ph=PhiN(tt),ph=phiN(tt);
  float mu=Ph/max(tt*Ph+ph,1e-4);
  float spreadV=mix(.66,1.,smoothstep(0.,2.5,tt));

  // sky reflection: four points of the visible slope distribution
  vec3 refl=vec3(0.); float Fm=0.;
  const float ZV[4]=float[4](-1.05,-.35,.35,1.05);
  const float ZP[4]=float[4](.75,-.75,.75,-.75);
  float spread=2.*sqrt(sv2*spreadV*spreadV*.35+sp2*.55);
  float lod=max(log2(spread/.00307),0.);
  for(int k=0;k<4;k++){
    vec2 sk=s-Vh*(mu+ZV[k]*spreadV)*sv+Vp*ZP[k]*sp;
    vec3 n=normalize(vec3(-sk.x,1.,-sk.y));
    float c=max(dot(V,n),.012);
    float F=fresnel(c);
    vec3 r=2.*c*n-V;
    refl+=F*envAt(r,lod);
    Fm+=F;
  }
  refl*=.25; Fm*=.25;
  refl=leadSea(refl,d);

  // sun: normal distribution of the slopes against the half vector
  vec3 glit=vec3(0.);
  vec3 L=uSun;
  vec3 Hv=normalize(V+L);
  vec2 sh=-Hv.xz/Hv.y;
  vec2 ds=sh-s;
  const float sunV=2.5e-6;
  float a=cov.x+sunV,b=cov.y+sunV,c=cov.z;
  float det=max(a*b-c*c,1e-9);
  float ex=(b*ds.x*ds.x-2.*c*ds.x*ds.y+a*ds.y*ds.y)/det;
  if(ex<140.){
    // real slopes are not quite Gaussian: most of the mass plus a long tail of steep facets
    const float K2=2.1;
    float P=(.945*exp(-.5*ex)+.055*exp(-.5*ex/(K2*K2))/(K2*K2))/(2.*PI*sqrt(det));
    vec2 Lh=normalize(L.xz);
    float sl2=max(dot(Lh,vec2(cov.x*Lh.x+cov.z*Lh.y,cov.z*Lh.x+cov.y*Lh.y)),1e-6);
    float zL=max(dot(L,N),.01);
    float zVc=max(zV,.01);
    float zH=Hv.y;
    float F=fresnel(dot(V,Hv));
    float Lmask=1.+lambdaS(zVc,sv2)+lambdaS(zL,sl2);
    glit=uSunE*F*P/(4.*zVc*zH*zH*zH*zH*Lmask);
    glit=min(glit,vec3(400.));
    // count it out into facets
    float cell=.00095*pow(2.,max(ceil(log2(fw*1.05/.00095)),0.));
    float Acell=tHit*tHit*cell*cell/grazing;
    const float Af=4.5e-4;                // m^2, effective glint facet at this sea state
    float Lone=dot(uSunL,vec3(.2126,.7152,.0722))*F*Af/Acell;
    float Lmean=dot(glit,vec3(.2126,.7152,.0722));
    float Ncell=Lmean/max(Lone,1e-4);
    // the rare facets far outside the path are small and only just catch the sun
    float cap=mix(2.5,70.,smoothstep(.02,.6,Ncell));
    if(uMode!=2){
      float sO=uHero<1.?sparkle(d,Ncell,cell):1.;
      float sG=uHero>0.?glints(d,Ncell,1.,fw):1.;
      glit=min(glit*mix(sO,sG,uHero),vec3(cap,cap,cap*.67));
    }
  }

  // water body, and the glow of low sun through the thin top of steep faces
  vec3 up=vec3(.0040,.0115,.0135)*(1.-Fm);
  float faces=sat(dot(N.xz,Vh)*7.);
  float crest=smoothstep(-.05,.55,hgt);
  float back=pow(sat(dot(d,normalize(vec3(L.x,0.,L.z)))),3.);
  vec3 sss=vec3(.075,.26,.17)*uSunE*.075*faces*crest*back*(1.-Fm);

  vec3 c0=refl+up+sss+glit;
  // air between the camera and the water
  float km=tHit*.001;
  vec3 Ta=exp(-uAir*km*vec3(.86,1.,1.22));
  vec2 pa=vec2(atan(d.x,d.z)/(2.*PI)+.5,.04);
  vec3 air=texture(uBg,pa).rgb;
  return c0*Ta+air*(1.-Ta);
}

#include "beach_surface.glsl"

// ------------------------------------------------------------------ the world loading
// returns rgba: a=0 final image from shade(), a=1 this stage overrides it
vec4 crackStage(vec3 d,float fw){
  float az=atan(d.x,d.z);
  if(az<=uCrack.z) return vec4(0.);
  float el=asin(d.y);
  vec2 g=vec2((az-uCrack.z)/uTile.x,(el-uHorEl)/uTile.y);
  vec2 tile=floor(g);
  float tl=uCrack.y+hash21(tile+vec2(37.,11.))*1.4;
  float dt=uT-tl;
  if(dt>=.55) return vec4(0.);
  ivec2 ig=ivec2(tile)+ivec2(0,uGrid.y/2);
  ig=clamp(ig,ivec2(0),uGrid-1);
  if(dt<0.){
    vec2 f=g-tile;
    vec2 e=min(f,1.-f)*uTile*vec2(cos(el),1.);
    float line=1.-smoothstep(fw*.5,fw*1.5,min(e.x,e.y));
    float horizonLine=1.-smoothstep(fw*.5,fw*1.5,abs(el-uHorEl));
    // at the film's exposure this is an untextured viewport: #2a2a2a with faint grid lines
    float v=mix(.062,.104,line);
    v=mix(v,.13,horizonLine);
    return vec4(vec3(v),1.);
  }
  if(dt<.25){
    vec3 c=texelFetch(uBlocks,ig,4).rgb;
    return vec4(c,1.);
  }
  vec2 f=g-tile;
  ivec2 bl=ig*4+ivec2(floor(f*4.));
  vec3 c=texelFetch(uBlocks,bl,2).rgb;
  // banded, as a low bit-depth proxy texture would be
  float lu=max(luma(c),1e-4);
  float k=pow(lu,.4545);
  const float Q=11.;
  k=floor(k*Q+.5)/Q;
  vec3 hue=c/lu;
  hue=floor(hue*7.+.5)/7.;
  return vec4(hue*pow(k,2.2),1.);
}

uniform vec2 uJit;      // where in the pixel this look lands, when a frame is several looks
void main(){
  vec2 v=(uv*2.-1.)*vec2(uAspect,1.)+uJit;
  vec3 d=viewDir(v);
  vec3 dx=dFdx(d),dy=dFdy(d);
  float fw=max(max(length(dx),length(dy)),1e-5)*uFoot;
  if(uCrack.x>.5){
    vec4 s=crackStage(d,fw);
    if(s.a>.5){ o=vec4(s.rgb*uGain,1.); return; }
  }
  o=vec4(shade(d,dx,dy,fw)*uGain,1.);
}
