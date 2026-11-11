#version 410
// The same rock as beach_stacks.frag, for the views facing the sun. That bake also goes into the sky and the side view
// and is left as it is. This one is lit: the forms, the rim the sun leaves on what faces sideways, the groups of columns
// in the lifted shadows, salt thickening towards the foot. No white at the foot: that moves, and lives in beach_mist.
in vec2 uv; out vec4 o;
#include "common.glsl"
const float PI=3.14159265;
const int NS=5;
uniform vec4 uRange;
uniform float uH,uCurv,uPix;
uniform vec3 uSun,uSunE;
uniform float uSkyLevel;
uniform sampler2D uBg;
uniform vec4 uS[NS];    // x,z,radius,height
uniform vec4 uS2[NS];   // elongation,angle,seed,column width

float sat(float x){ return clamp(x,0.,1.); }

float body(vec2 p,int i){
  vec4 a=uS[i],b=uS2[i];
  vec2 q=rot(b.y)*(p-a.xy)/(a.z*vec2(b.x,1.));
  // cliffs all round but never plumb: buttresses, notches, and a shoulder lower on one side
  float l=length(q)+.6*(fbm(p/(a.z*.75)+b.z,3)-.44)+.2*(fbm(p/(a.z*.17)+b.z+3.,3)-.42);
  float taper=pow(sat(1.-l),.62);
  float crown=fbm(p/(a.z*.6)+b.z+7.,4);
  float h=a.w*taper*(.36+1.45*max(crown-.2,0.))*1.25;
  float skirt=a.w*.05*smoothstep(1.14,.95,l)*(.2+1.6*fbm(p/(a.z*.25)+b.z+11.,2));
  return max(h,skirt);
}
float headland(vec2 p,float az,float r){
  if(az<.3) return -1.;
  float rc=2450.-620.*sin(sat((az-.3)*.95)*1.57)+260.*(fbm(vec2(az*5.,1.3),3)-.45);
  float x=(r-rc)/330.;
  float nose=az+.05*(fbm(vec2(r/260.,4.),3)-.4);
  float mask=smoothstep(.4,.455,nose)*(1.-smoothstep(1.32,1.46,az));
  // old lava flows, one on another: a skyline of shoulders and gullies, not a dome
  float gully=1.-abs(2.*fbm(p/120.+3.,3)-1.);
  float crest=(40.+170.*fbm(vec2(az*6.5,r/900.),4)+70.*(fbm(p/62.,4)-.42)+64.*(gully-.62)+9.*smoothstep(.3,.7,fract(fbm(p/90.+9.,2)*7.)))*mask;
  float cliff=smoothstep(-.3,-.1,x+.25*(fbm(p/170.,3)-.4));
  float back=1.-smoothstep(.3,2.4,x);
  return crest*cliff*back;
}

float field(vec2 p,int i){ return i<NS-1?body(p,i):headland(p,atan(p.x,p.y),length(p)); }

struct Col{ float h; vec2 id; vec2 c; };
Col column(vec2 p,int i){
  float cs=uS2[i].w;
  vec2 g=p/cs,c=floor(g);
  float best=9.; vec2 bid=c,bctr=c;
  for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++){
    vec2 id=c+vec2(float(x),float(y));
    vec2 ctr=id+.12+.76*hash22(id+uS2[i].z);
    vec2 r=ctr-g; float d=dot(r,r);
    if(d<best){ best=d; bid=id; bctr=ctr; }
  }
  Col k; k.id=bid; k.c=bctr*cs;
  float b=i<NS-1?body(k.c,i):0.;
  if(i==NS-1){ float rc=length(k.c); b=headland(k.c,atan(k.c.x,k.c.y),rc); }
  float jit=(hash21(bid+uS2[i].z*3.1)-.5)*(i<NS-1?.02*uS[i].w:2.6);
  k.h=b>.3?max(b+jit*smoothstep(.0,.2*max(uS[i].w,60.),b),.2):-1.;
  return k;
}

// one view ray: returns radiance (premultiplied by hit), .a=hit
vec4 ray(vec2 pix){
  float az=mix(uRange.x,uRange.y,pix.x),el=mix(uRange.z,uRange.w,pix.y);
  float tanEl=tan(el);
  vec2 dir=vec2(sin(az),cos(az));
  float bestT=1e9; vec4 res=vec4(0.);
  for(int i=0;i<NS;i++){
    float t0,t1,dt;
    if(i<NS-1){
      vec2 c=uS[i].xy;
      float R=uS[i].z*max(uS2[i].x,1.)*1.75;
      float m=dot(c,dir);
      float off=length(c-dir*m);
      if(off>R) continue;
      float hw=sqrt(R*R-off*off);
      t0=m-hw; t1=m+hw;
      dt=uS2[i].w*.2;
      // above everything it could reach
      if(uH+t1*tanEl-t1*t1*uCurv>uS[i].w*1.2&&tanEl>0.&&uH+t0*tanEl-t0*t0*uCurv>uS[i].w*1.2) continue;
    } else {
      if(az<.3||az>1.5) continue;
      t0=1450.; t1=3600.; dt=1.4;
      if(uH+t0*tanEl-t0*t0*uCurv>280.) continue;
    }
    if(t0>=bestT) continue;
    for(float t=t0;t<min(t1,bestT);t+=dt){
      vec2 p=dir*t;
      float y=uH+t*tanEl-t*t*uCurv;
      if(y<0.) break;
      Col k=column(p,i);
      if(y<=k.h){
        bestT=t;
        // The columns are under a pixel out here, the buttresses are not: the light goes by the big forms.
        vec3 V=normalize(vec3(dir.x,tanEl,dir.y));
        float e=i<NS-1?2.5:9.;
        vec2 g=vec2(field(k.c+vec2(e,0.),i)-field(k.c-vec2(e,0.),i),field(k.c+vec2(0.,e),i)-field(k.c-vec2(0.,e),i));
        vec3 Nb=normalize(vec3(-g.x,2.*e,-g.y));
        // A pattern in plan comes out on a wall as upright strips: groups of columns, each turned a little its own way.
        float cw=i<NS-1?3.1:13.;
        float s1=vnoise(k.c/cw+uS2[i].z),s2=vnoise(k.c/(cw*.38)+uS2[i].z+5.);
        float strip=s1*.68+s2*.32;
        vec2 tang=vec2(-Nb.z,Nb.x)/max(length(Nb.xz),.05);
        vec3 N=normalize(Nb+vec3(tang.x,0.,tang.y)*(strip-.5)*2.6);
        // and they break off in steps, where a set of them stops and the next begins
        float ledge=vnoise(vec2(floor(strip*11.)*3.7,y/(i<NS-1?5.2:22.)));
        float hs=hash21(k.id+1.7);
        float alb=(.05+.065*s2+.028*hs)*(.72+.56*ledge);
        // wet and weed-black from the tide up, not ruled off
        alb*=mix(.48,1.,smoothstep(1.2,8.,y+4.*(s1-.5)));
        // what the rock can see of the sun, almost level and behind it: only what faces sideways and stands clear
        vec2 Lh=normalize(uSun.xz);
        float tl=uSun.y/length(uSun.xz),sunV=1.,stp=i<NS-1?uS[i].z*.2:36.;
        for(int s=1;s<=12;s++){
          float ds=float(s)*stp;
          sunV=min(sunV,smoothstep(-1.5,1.5,y+ds*tl-field(k.c+Lh*ds,i)));
        }
        // the shadow side faces the cold half of the sky, and the bright air round the rock helps a little
        vec3 fill=vec3(.3,.33,.41)*(.8+.3*Nb.y+.12*dot(Nb.xz,-Lh));
        vec3 c=alb*fill+alb*uSunE*max(dot(N,uSun),0.)*sunV*(i<NS-1?.85:.3);
        // wet rock taken edge on is a mirror of whatever is beyond it, and beyond it is the low bright sky
        vec3 Rf=reflect(V,N);
        float Fr=.04+.96*pow(1.-max(dot(N,-V),0.),5.);
        vec3 glow=texture(uBg,vec2(atan(Rf.x,Rf.z)/(2.*PI)+.5,.05)).rgb;
        vec3 air=texture(uBg,vec2(az/(2.*PI)+.5,.05)).rgb;
        // only on the side the sun is: the other edge looks at the same sky through a rock's own shade
        vec2 toSun=vec2(dir.y,-dir.x)*sign(dir.y*Lh.x-dir.x*Lh.y);
        float sideS=dot(N.xz,toSun);
        // (the headland stands far back and its columns come to a pixel: on it this only gives away the blocks)
        float rk=i<NS-1?1.:.22;
        c+=min(glow*Fr*.3*sunV*smoothstep(-.15,.55,sideS),air*.42)*rk;
        // rough stone lit and seen nearly edge on still throws the light on forward
        c+=alb*uSunE*sunV*smoothstep(.25,.95,sideS)*.55*rk;
        // the air between goes by range, and low down it is thick with salt: the foot goes before the top does
        float Ta=exp(-t*.001*.115)*exp(-t*.00034*exp(-y/10.));
        // cold wet air takes the colour out of what it covers before it lifts it
        vec3 veil=mix(air,vec3(dot(air,vec3(.25,.65,.1)))*vec3(.96,.99,1.04),.75);
        c=c*Ta+veil*(1.-Ta);
        res=vec4(c,1.);
        break;
      }
    }
  }
  return res;
}

void main(){
  vec2 px=1./vec2(textureSize(uBg,0))*0.+uPix*vec2(1./(uRange.y-uRange.x),1./(uRange.w-uRange.z));
  vec4 acc=vec4(0.);
  const vec2 SS[4]=vec2[4](vec2(-.375,-.125),vec2(.125,-.375),vec2(.375,.125),vec2(-.125,.375));
  for(int i=0;i<4;i++) acc+=ray(uv+SS[i]*px);
  o=acc*.25;
}
