#version 410
// one drop, hanging. a ray comes through one piece of the lens, meets the water, and is followed:
// some of it off the front, the rest through, out of the back or around inside first,
// or up against the underside of the petal. nothing here knows what the picture in a drop looks like.
in vec2 uv; out vec4 o;
uniform vec3 uC,uF,uR,uU;
uniform vec2 uLens,uGc,uJit,uRes;
uniform float uFd,uTan,uZoom,uAspect,uW;
// the drop: its frame, its size, how far it is stretched
uniform vec3 uDc,uDx,uDy,uDz;
uniform float uDr,uRv,uPxV,uRhoS,uYp;
uniform vec2 uSt;
uniform sampler2D uProf;
uniform vec3 uProfW;
uniform vec2 uBound;
// the petal it hangs from
uniform sampler2D uSheet,uLitS,uPigS;
uniform vec3 uSheetW;
uniform float uLodS;
// the map of the sheet is drawn for the whole drop. while less of it has got here, a point of it is somewhere else on the map
uniform float uMapS;
uniform vec3 uMapO;
// what is out there
uniform sampler2D uAtlas,uBehind,uFarW,uFarH,uEnv;
uniform vec3 uS0x,uS0y,uS0z,uFw,uRw,uUw,uFh,uRh,uUh,uFe,uRe,uUe;
uniform float uTanW,uTanH,uTanE,uTanC,uTanB,uPxC,uPxB,uAspS,uHasH,uFarK,uFarMax,uMorph,uProbe;
const float PI=3.14159265;
const float N1=1.333;
const float PX_S=477.;

// ---- shape -----------------------------------------------------------------------------------
vec2 meridian(vec3 p){
  return vec2(length(p.xz)/uSt.x,uYp+(p.y-uYp)/uSt.y);
}
vec4 prof(vec2 q){
  return texture(uProf,vec2(q.x/uProfW.x,(q.y-uProfW.y)/uProfW.z));
}
float mapD(vec3 p){
  return prof(meridian(p)).x*min(uSt.x,uSt.y);
}
vec3 normalAt(vec3 p){
  vec2 q=meridian(p);
  vec4 s=prof(q);
  float r=max(length(p.xz),1e-5);
  return normalize(vec3(s.y/uSt.x*p.x/r,s.z/uSt.y,s.y/uSt.x*p.z/r));
}
vec3 toWorld(vec3 v){ return v.x*uDx+v.y*uDy+v.z*uDz; }

// ---- the sheet ---------------------------------------------------------------------------------
vec4 sheetAt(vec2 xz){
  return texture(uSheet,xz/(2.*uSheetW.x)+.5);
}
bool inMap(vec2 xz){ return abs(xz.x)<uSheetW.x&&abs(xz.y)<uSheetW.x; }
bool solid(vec2 st){
  return st.x>0.&&st.x<1.&&st.y>0.&&st.y<1.&&textureLod(uPigS,st,1.).a>.5;
}
// does this leg of the ray meet the petal?
bool sheetCross(vec3 p,vec3 d,float t0,float t1,out float ts,out vec2 st){
  ts=0.; st=vec2(0.);
  p=p*uMapS+uMapO; d*=uMapS;
  float a=t0,b=t1;
  if(abs(d.y)>1e-5*uMapS){
    float ta=(uSheetW.y-p.y)/d.y,tb=(uSheetW.z-p.y)/d.y;
    a=max(a,min(ta,tb)); b=min(b,max(ta,tb));
  } else if(p.y<uSheetW.y||p.y>uSheetW.z) return false;
  if(a>=b) return false;
  const int N=12;
  float dt=(b-a)/float(N);
  bool have=false;
  float prev=0.,tp=a;
  for(int i=0;i<=N;i++){
    float t=a+dt*float(i);
    vec3 q=p+d*t;
    if(!inMap(q.xz)){ have=false; continue; }
    vec4 s=sheetAt(q.xz);
    if(s.a<.999){ have=false; continue; }
    float f=q.y-s.x;
    if(have&&f*prev<=0.){
      float l=tp,h=t,fl=prev;
      for(int k=0;k<5;k++){
        float m=.5*(l+h);
        vec3 qm=p+d*m;
        float fm=qm.y-sheetAt(qm.xz).x;
        if(fm*fl<=0.) h=m; else { l=m; fl=fm; }
      }
      vec3 qh=p+d*h;
      vec4 sh=sheetAt(qh.xz);
      if(solid(sh.yz)){ ts=h; st=sh.yz; return true; }
    }
    prev=f; tp=t; have=true;
  }
  return false;
}

// ---- out there -------------------------------------------------------------------------------
vec4 cube(vec3 d,float lod){
  vec3 a=abs(d);
  vec2 sc,tile;
  if(a.x>=a.y&&a.x>=a.z){ sc=d.x>0.?vec2(-d.z,d.y)/a.x:vec2(d.z,d.y)/a.x; tile=d.x>0.?vec2(0.,0.):vec2(1.,0.); }
  else if(a.y>=a.z){ sc=d.y>0.?vec2(d.x,-d.z)/a.y:vec2(d.x,d.z)/a.y; tile=d.y>0.?vec2(2.,0.):vec2(0.,1.); }
  else { sc=d.z>0.?vec2(d.x,d.y)/a.z:vec2(-d.x,d.y)/a.z; tile=d.z>0.?vec2(1.,1.):vec2(2.,1.); }
  return textureLod(uAtlas,(tile+sc/uTanC*.5+.5)/vec2(3.,2.),lod);
}
float gOut=0.;   // how far out of focus the near things have gone, in mip levels
vec4 flora(vec3 d,float fp){
  float lc=log2(max(fp*uPxC,1.));
  vec4 c=cube(d,min(lc,3.5)+gOut);
  if(d.z>0.){
    vec2 v=d.xy/d.z/uTanB;
    float e=max(abs(v.x),abs(v.y));
    if(e<1.){
      vec4 b=textureLod(uBehind,v*.5+.5,log2(max(fp*uPxB,1.))+gOut);
      c=mix(c,b,smoothstep(1.,.93,e));
    }
  }
  return c;
}
vec3 skyAll(vec3 ds,vec3 dw,float lod){
  float az=atan(ds.x,ds.z);
  vec3 c;
  if(ds.y>=0.) c=textureLod(uEnv,vec2(az/(2.*PI)+.5,asin(min(ds.y,1.))/(.5*PI)),lod).rgb;
  else {
    vec3 h=textureLod(uEnv,vec2(az/(2.*PI)+.5,.004),lod).rgb;
    c=mix(vec3(.008,.008,.009),h*.4,exp(ds.y/.012)*.8);
  }
  // the brightest of the cloud is behind whoever is holding the camera, up and to the left: every other drop has it too
  float tc=length(dw-vec3(-.46,.62,-.63));
  c+=vec3(.92,.96,1.)*9.*exp(-tc*tc/(2.*.17*.17));
  return c;
}
// through the sea's own camera. a pinhole sees part of the sphere: past that, the sky it has baked
vec3 far(vec3 ds,vec3 dw,float fp){
  vec3 c;
  float a=0.;
  float z=dot(ds,uFw);
  if(z>0.){
    vec2 v=vec2(dot(ds,uRw),dot(ds,uUw))/z/uTanW/vec2(uAspS,1.);
    a=smoothstep(1.,.96,max(abs(v.x),abs(v.y)));
    if(a>0.) c=textureLod(uFarW,v*.5+.5,log2(max(fp*PX_S/uTanW,1.))).rgb;
  }
  vec3 s=vec3(0.);
  if(a<1.) s=skyAll(ds,dw,clamp(log2(max(fp*80.,1.)),2.,6.));
  c=a>0.?mix(s,min(c*uFarK,vec3(uFarMax)),a):s;
  if(uHasH>.5){
    z=dot(ds,uFh);
    if(z>0.){
      vec2 v=vec2(dot(ds,uRh),dot(ds,uUh))/z/uTanH/vec2(uAspS,1.);
      float e=max(abs(v.x),abs(v.y));
      float b=smoothstep(1.,.96,e);
      if(b>0.) c=mix(c,min(textureLod(uFarH,v*.5+.5,log2(max(fp*PX_S/uTanH,1.))).rgb*uFarK,vec3(uFarMax)),b);
    }
  }
  return c;
}
vec3 toSea(vec3 d){ return d.x*uS0x+d.y*uS0y+d.z*uS0z; }
vec3 look(vec3 dw,float fp){
  vec4 f=flora(dw,fp);
  if(f.a>=.999) return f.rgb;
  return f.rgb+far(toSea(dw),dw,fp)*(1.-f.a);
}
float fres(float c){
  float g=sqrt(max(N1*N1-1.+c*c,0.));
  float a=(g-c)/(g+c);
  float b=(c*(g+c)-1.)/(c*(g-c)+1.);
  return .5*a*a*(1.+b*b);
}

// ---- inside ------------------------------------------------------------------------------------
vec3 inside(vec3 p,vec3 d,float fp,vec3 dIdeal){
  vec3 acc=vec3(0.),thr=vec3(1.);
  float wk=1.-uMorph;
  for(int bounce=0;bounce<5;bounce++){
    float t=.0015;
    for(int i=0;i<56;i++){
      float dd=-mapD(p+d*t);
      if(dd<.0003) break;
      t+=dd;
    }
    float ts; vec2 st;
    if(sheetCross(p,d,0.,t,ts,st)){
      // the underside of the sheet, lit from above it: it is what the sun has already been through
      vec3 l=textureLod(uLitS,st,uLodS+float(bounce)+gOut).rgb;
      acc+=thr*l*wk;
      return acc;
    }
    vec3 h=p+d*t;
    vec3 n=normalAt(h);
    float ci=max(dot(d,n),0.);
    float s2=N1*N1*(1.-ci*ci);
    float Fi=1.;
    if(s2<1.){
      float ct=sqrt(1.-s2);
      Fi=fres(ct);
      vec3 dout=normalize(N1*d+(ct-N1*ci)*n);
      vec3 dw=toWorld(dout);
      if(uProbe>.5) return toSea(dw);
      float f2=fp*(1.+float(bounce));
      if(bounce==0&&uMorph>0.){
        // the camera comes to rest on the sea. focus goes out past the flower: it leaves as a blur, not as a ghost
        vec3 ds=normalize(mix(toSea(dw),dIdeal,uMorph));
        vec4 fl=flora(dw,f2)*(1.-smoothstep(.1,.85,uMorph));
        acc+=thr*(1.-Fi*wk)*(fl.rgb+far(ds,dw,f2)*(1.-fl.a));
      } else {
        acc+=thr*(1.-Fi)*look(dw,f2)*(bounce==0?1.:wk);
      }
    }
    thr*=Fi*wk;
    if(max(thr.x,max(thr.y,thr.z))<.015) return acc;
    d=d-2.*ci*n;
    p=h;
    fp*=1.6;
  }
  return acc+thr*look(toWorld(d),.05);
}
float hash(vec2 p){ return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453); }

void main(){
  gOut=5.5*smoothstep(0.,.7,uMorph);
  vec2 ndc0=gl_FragCoord.xy/uRes*2.-1.;
  vec2 ndc=ndc0-uJit;
  vec2 view=ndc*vec2(uAspect,1.)/uZoom+uGc;
  float s=1./uFd;
  vec3 Pf=uC+uF*s+(uR*view.x+uU*view.y)*uTan*s;
  // the same pattern on the lens, turned differently for each pixel: grain rather than copies
  float ang=hash(gl_FragCoord.xy)*6.2831853;
  float ca=cos(ang),sa=sin(ang);
  vec2 L=vec2(ca*uLens.x-sa*uLens.y,sa*uLens.x+ca*uLens.y);
  vec3 O=uC+uR*L.x+uU*L.y;
  vec3 dW=normalize(Pf-O);
  vec3 r0=(O-uDc);
  vec3 rd=vec3(dot(dW,uDx),dot(dW,uDy),dot(dW,uDz));
  vec3 ro=vec3(dot(r0,uDx),dot(r0,uDy),dot(r0,uDz))/uDr;
  vec3 bc=vec3(0.,uBound.x,0.);
  // start beside it: thirty centimetres is a hundred radii, and a float only has so much to give
  ro+=rd*dot(bc-ro,rd);
  float disc=uBound.y*uBound.y-dot(ro-bc,ro-bc);
  if(disc<=0.) discard;
  float sq=sqrt(disc);
  float t=-sq;
  bool hit=false;
  for(int i=0;i<72;i++){
    float dd=mapD(ro+rd*t);
    if(dd<.0005){ hit=true; break; }
    t+=dd;
    if(t>sq) break;
  }
  if(!hit) discard;
  t+=mapD(ro+rd*t);
  t+=mapD(ro+rd*t);
  vec3 p=ro+rd*t;
  // the petal in front of it? then it can't be seen from here
  float ts; vec2 st;
  if(sheetCross(ro,rd,-sq,t,ts,st)) discard;
  vec3 pm=p*uMapS+uMapO;
  if(inMap(pm.xz)){
    vec4 sh=sheetAt(pm.xz);
    if(sh.a>.999&&pm.y>sh.x&&solid(sh.yz)) discard;
  }
  vec3 n=normalAt(p);
  float c0=max(dot(-rd,n),0.);
  float F=fres(c0);
  float h0=sqrt(max(1.-c0*c0,0.));
  // how far one pixel reaches, out in the world
  float drho=1./(uPxV*uRv);
  float da=2.*(1./sqrt(max(1.-h0*h0,.012))-1./sqrt(N1*N1-h0*h0));
  float fp=da*(drho+uRhoS);
  // the frame this pixel will have when the sea takes over: upside down
  vec2 fv=ndc0*vec2(uAspect,1.);
  vec3 dIdeal=normalize(uFe-fv.x*uTanE*uRe-fv.y*uTanE*uUe);
  vec3 dt=normalize(rd/N1+(c0/N1-sqrt(max(1.-(1.-c0*c0)/(N1*N1),0.)))*n);
  vec3 c=inside(p,dt,fp,dIdeal);
  if(uProbe>.5){ o=vec4(c,1.); return; }
  vec3 rv=reflect(rd,n);
  float wk=1.-uMorph;
  vec3 ref=look(toWorld(rv),(drho+uRhoS)*2./max(c0,.12));
  c=c*(1.-F*wk)+ref*F*wk;
  o=vec4(c,1.)*uW;
}
